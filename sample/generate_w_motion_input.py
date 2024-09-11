# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader, get_collate_fn
from data_loaders.humanml.scripts.motion_process import recover_from_ric, compute_uncertainty_factor
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.plot_script import plot_3d_motion_with_gt
import shutil
from data_loaders.tensors import collate
from diffusion.losses import calculate_ause, discretized_gaussian_log_likelihood
from diffusion.nn import mean_flat, sum_flat
import matplotlib.pyplot as plt


def main():
    args = generate_args()
    print(f'Args: %s' % args)
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                            #   hml_mode='train') # in train mode, you get both text and motion.
                              hml_mode='eval')  
    total_num_samples = args.num_samples * args.num_repetitions

    # Use the subset data loader
    print("Extracting sequences from the end of the dataset...")
    total_samples = len(data.dataset)
    start_index = total_samples - 2300
    subset_indices = list(range(start_index, total_samples))
    subset_dataset = torch.utils.data.Subset(data.dataset, subset_indices)
    subset_data_loader = torch.utils.data.DataLoader(subset_dataset, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=False, 
                                                    num_workers=8, 
                                                    collate_fn=get_collate_fn(args.dataset, 'eval'))


    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(subset_data_loader) # for fixed long sequences to compare different models
        # iterator = iter(data) #for random sequences from the evaluation part of the dataset
        input_motions, model_kwargs = next(iterator)
        input_motions = input_motions.to(dist_util.dev())
        # print(model_kwargs['y']['text'])
        # raise SystemExit
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        input_motions, model_kwargs = collate(collate_args)
        input_motions = input_motions.to(dist_util.dev())

    all_motions = []
    all_variances = []
    all_mean_fluctuations = []
    all_lengths = []
    all_text = []
    per_joint_errors_all = []

    start_idx = args.emb_motion_len

    # # For motion editing
    # model_kwargs['y']['inpainted_motion'] = input_motions
    # # print(f'Input motion shape: {input_motions.shape}') # [bs, njoints, 1, seqlen]
    # model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
    #                                                         device=input_motions.device)  # True means use gt motion
    # for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
    #     model_kwargs['y']['inpainting_mask'][i, :, :, 50:] = False  # do inpainting in those frames

    times_ms = [0, 0.5, 1, 1.5, 2, 2.45, 2.95, 3.45, 3.95, 4.45, 4.95, 5.45, 5.95, 6.45, 6.95, 7.45, 7.95]
    frame_indices = [int(20 * t) for t in times_ms]
    mpjpe_specific_times = [[] for _ in times_ms]  # List to store MPJPE at specific times


    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        model_kwargs['y']['motion_embed'] = input_motions
        model_kwargs['y']['motion_embed_mask'] = torch.ones_like(input_motions, dtype=torch.bool, device=input_motions.device)
        model_kwargs['y']['motion_embed_mask'][:, :, :, start_idx:] = False
        sample_fn = diffusion.p_sample_loop

        if args.learning_var:
            sample, log_variance, mean_fluctuations = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        else:
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )            

        # print(input_motions.cpu().shape) # [10, 263, 1, 196]
        # print(sample.cpu().shape) # [10, 263, 1, 196]
        # print(log_variance.cpu().shape) # [10, 263, 1, 196]

        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21 # sample.shape = [bs, 263, 1, 196]
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float() # [bs, 1, 196, 263]
            sample = recover_from_ric(sample, n_joints) # [bs, 1, 196, 22, 3]
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) # [bs, 22, 3, 196]
            if args.learning_var:
                log_variance = data.dataset.t2m_dataset.inv_transform(log_variance.cpu().permute(0, 2, 3, 1)).float() # [bs, 1, 196, 263]
                uncertainty_factor = compute_uncertainty_factor(log_variance, n_joints) # [bs, 22, 196]

                mean_fluctuations = mean_fluctuations.cpu().permute(0, 2, 3, 1).float() # [bs, 1, 196, 263]
                uncertainty_factor_1 = compute_uncertainty_factor(mean_fluctuations, n_joints) # [bs, 1, 196, 22]

            input_motions_reshaped = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
            input_motions_reshaped = recover_from_ric(input_motions_reshaped, n_joints)
            input_motions_reshaped = input_motions_reshaped.view(-1, *input_motions_reshaped.shape[2:]).permute(0, 2, 3, 1)

        # Applying rot2xyz transformation
        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False) # [10, 22, 3, 196]
        
        input_motions_reshaped = model.rot2xyz(x=input_motions_reshaped, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                  jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                  get_rotations_back=False)
        
        valid_frame_mask = torch.zeros_like(input_motions_reshaped, dtype=torch.bool) # [bs, 22, 3, 196]

        for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
            valid_frame_mask[i, :, :, :length] = 1 # [64, 22, 3, 196]
            sample[i, :, :, length:] = 0
            input_motions_reshaped[i, :, :, length:] = 0

        # Compute MPJPE
        target_xyz_reshaped = input_motions_reshaped.permute(0, 3, 1, 2).reshape(args.batch_size, -1, 3) # torch.size([bs, 196*22, 3])
        pred_xyz_reshaped = sample.permute(0, 3, 1, 2).reshape(args.batch_size, -1, 3) # torch.size([bs, 196*22, 3])
        per_joint_errors = torch.norm(target_xyz_reshaped - pred_xyz_reshaped, 2, 2) # torch.Size([bs, 196*22])
        per_joint_errors_all.append(per_joint_errors)

        errors_reshaped = per_joint_errors.mean(dim=0) # Mean over batch #torch.Size([196*22])
        overtime_3d_err = errors_reshaped.view(-1, n_joints).mean(dim=1) # torch.Size([196])

        # Compute MPJPE at specific time frames
        for t_idx, frame_idx in enumerate(frame_indices):
                valid_mask_at_frame = valid_frame_mask[:, :, :, frame_idx+10].any(dim=1).any(dim=0) # +10 is added because there is variability in the lengths of sequences within the same batch, so we prefer not to consider the last 10 frames to avoid this irregularity
                if valid_mask_at_frame.any().item():  
                    mpjpe_at_time = overtime_3d_err[frame_idx]
                    mpjpe_specific_times[t_idx].append(mpjpe_at_time.item())
                    # print(f'Repetition {rep_i} - Time {times_ms[t_idx]}s - MPJPE: {mpjpe_at_time*1000:.4f}')

        # Compute AUSE
        if args.learning_var:
            uncertainty_factor_ause = uncertainty_factor.squeeze(1)
            uncertainty_factor1_ause = uncertainty_factor_1.squeeze(1)
            sparsification_errors_lg, oracle, sparsification_levels_lg = calculate_ause(per_joint_errors, uncertainty_factor_ause, model_kwargs['y']['lengths'])
            sparsification_errors_mf, oracle, sparsification_levels_mf = calculate_ause(per_joint_errors, uncertainty_factor1_ause, model_kwargs['y']['lengths'])

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        if args.learning_var:
            all_variances.append(uncertainty_factor.cpu().numpy())
            all_mean_fluctuations.append(uncertainty_factor_1.cpu().numpy())
        # print(f"created {len(all_motions) * args.batch_size} samples")

    # mpjpe = sum(mean_errors) / len(mean_errors) if mean_errors else float('inf')
    # print(f'---> Overall MPJPE = {mpjpe*1000:.4f}')
    # print(times_ms)
    for time_ms, errors_at_time in zip(times_ms, mpjpe_specific_times):
        if errors_at_time:
            avg_error = sum(errors_at_time) / len(errors_at_time)
            print(f'---> MPJPE at {time_ms} s = {avg_error*1000:.4f}')
    
    all_motions = np.concatenate(all_motions, axis=0) # [num_samples*num_repetitions, njoints, 3, seqlen]
    # all_motions = all_motions[:total_num_samples]  # [num_samples*num_repetitions, njoints, 3, seqlen]
    
    # Compute uncertainty factor based on standard deviation between repetitions
    all_motions_tensor = torch.from_numpy(all_motions)
    uncertainty_particle = []
    for sample_i in range(args.num_samples):
        sample_motions = all_motions_tensor[sample_i::args.num_samples]
        std_dev = torch.std(sample_motions, dim=0, keepdim=True)
        std_dev = std_dev.repeat(args.num_repetitions, 1, 1, 1)  # [num_samples*num_repetitions, njoints, 3, seqlen]
        uncertainty_particle.append(std_dev)
    
    uncertainty_particle = torch.stack(uncertainty_particle, dim=0)  # [num_samples, num_repetitions, njoints, 3, seqlen]
    uncertainty_particle = uncertainty_particle.permute(1, 0, 2, 3, 4)  # [num_repetitions, num_samples, njoints, 3, seqlen]
    uncertainty_particle = uncertainty_particle.reshape(-1, *uncertainty_particle.shape[2:])  # [num_samples*num_repetitions, njoints, 3, seqlen]
    uncertainty_particle = uncertainty_particle.permute(0, 2, 3, 1)  # [num_samples*num_repetitions, 3, seqlen, njoints]

    # Compute AUSE for uncertainty_particle
    per_joint_errors_all = torch.cat(per_joint_errors_all, dim=0)  # [num_samples*num_repetitions, 196*22]

    uncertainty_particle_ause = uncertainty_particle.mean(dim=1) # [num_repetitions*num_samples, seqlen, njoints]
    sparsification_errors_up, oracle, sparsification_levels_up = calculate_ause(per_joint_errors_all, uncertainty_particle_ause, model_kwargs['y']['lengths'])
    plt.figure(figsize=(10, 6))
    # plt.plot(sparsification_levels_lg, sparsification_errors_lg, marker='o', linestyle='-', color='b', label='Predicted Variance')
    # plt.plot(sparsification_levels_mf, sparsification_errors_mf, marker='s', linestyle=':', color='b', label='Denoising Fluctuations')
    plt.plot(sparsification_levels_up, sparsification_errors_up, marker='s', linestyle='--', color='b', label='Mode Divergence')
    plt.plot(sparsification_levels_up, oracle, marker='s', linestyle='--', color='g', label='Oracle')
    plt.xlabel('Sparsification Level (Fraction of Data Removed)')
    plt.ylabel('Sparsification Error')
    plt.title('Sparsification Error vs. Sparsification Level')
    plt.legend()
    plt.grid(True)
    plt.savefig('sparsification_error_plot_uncertainty_particle.png')
    plt.close()
    
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    if args.learning_var:
        all_variances = np.concatenate(all_variances, axis=0)
        all_variances = all_variances[:total_num_samples] # [num_samples*num_repetitions, 1, seqlen, njoints]
        all_mean_fluctuations = np.concatenate(all_mean_fluctuations, axis=0)
        all_mean_fluctuations = all_mean_fluctuations[:total_num_samples] # [num_samples*num_repetitions, 1, seqlen, njoints]
    all_uncertainty_particle = uncertainty_particle.cpu().numpy()
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    # if args.learning_var:
    np.save(npy_path,
        {'motion': all_motions, 'variances': all_uncertainty_particle, 'text': all_text, 'lengths': all_lengths,
            'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    # else:
    #     np.save(npy_path,
    #         {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
    #          'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            input_motions_reshaped_np = input_motions_reshaped.cpu().detach().numpy()  # if needed
            input_motion_reshaped = input_motions_reshaped_np[sample_i].transpose(2, 0, 1)[:length]
            # print(f"motion shape: {motion.shape}") (196, 22, 3)
            if args.learning_var:
                variance = all_variances[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]  # [njoints, 1, seqlen]
                # Apply smoothing to mean_fluctuation over time
                window_size = 3  # Adjust this value to control the amount of smoothing
                mean_fluctuation = all_mean_fluctuations[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length] # [seqlen, njoints-1, 3]
            particle_uncertainty = all_uncertainty_particle[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length] # [seqlen, njoints, 3]
                # mean_fluctuation_smoothed = np.zeros_like(mean_fluctuation)
                # for i in range(mean_fluctuation.shape[0]):  # Iterate over joints
                #     for j in range(mean_fluctuation.shape[1]):  # Iterate over dimensions
                #         mean_fluctuation_smoothed[i, j, :] = np.convolve(mean_fluctuation[i, j, :], np.ones(window_size)/window_size, mode='same')
                # mean_fluctuation = mean_fluctuation_smoothed

            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            # plot_3d_motion(animation_save_path, skeleton, motion, variance=variance, dataset=args.dataset, title=caption, fps=fps) #modified plot_3d_motion to include variance
            assert motion.shape[0] == input_motion_reshaped.shape[0], f"Frame mismatch: joints has {motion.shape[0]} frames, gt_data has {input_motions_reshaped.shape[0]} frames."
            # if args.learning_var:
                # plot_3d_motion_with_gt(animation_save_path, skeleton, motion, dataset=args.dataset, variance=variance, gt_data=input_motion_reshaped, title=caption, fps=fps, emb_motion_len=args.emb_motion_len) #modified plot_3d_motion to include gt input motions
            plot_3d_motion_with_gt(animation_save_path, skeleton, motion, dataset=args.dataset, variance=particle_uncertainty, gt_data=input_motion_reshaped, title=caption, fps=fps, emb_motion_len=args.emb_motion_len) #modified plot_3d_motion to include gt input motions
            # else:
            #     plot_3d_motion_with_gt(animation_save_path, skeleton, motion, dataset=args.dataset, gt_data=input_motion_reshaped, title=caption, fps=fps, emb_motion_len=args.emb_motion_len) #modified plot_3d_motion to include gt input motions

            
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(args, out_path,
                                               row_print_template, all_print_template, row_file_template, all_file_template,
                                               caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    # print(args.dataset)
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
