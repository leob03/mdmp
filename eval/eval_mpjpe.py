"""
Generate a large batch of samples and evaluate the whole test set over MPJPE.
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
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.plot_script import plot_3d_motion_with_gt
import shutil
from data_loaders.tensors import collate
from tqdm import tqdm


def main():
    args = generate_args()
    print("args.lv: ", args.lv)
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

    print('Loading dataset...')
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                            #   hml_mode='train') # in train mode, you get both text and motion.
                              hml_mode='eval')  

    total_samples = len(data.dataset)
    # print(total_samples) #4646


    # nb_samples_longer_than_three_seconds = 4328
    nb_samples_longer_than_three_seconds = 4288
    start_index = total_samples - nb_samples_longer_than_three_seconds
    subset_indices = list(range(start_index, total_samples))
    subset_dataset = torch.utils.data.Subset(data.dataset, subset_indices)
    subset_data_loader = torch.utils.data.DataLoader(subset_dataset, 
                                                    batch_size=args.batch_size, 
                                                    shuffle=False, 
                                                    num_workers=8, 
                                                    collate_fn=get_collate_fn(args.dataset, 'eval'))

    total_batches = len(subset_data_loader.dataset)
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    all_motions = []
    all_variances = []
    all_lengths = []
    all_text = []

    start_idx = args.emb_motion_len # 50 in most of our experiments
    start_time_ms = start_idx / 20

    times_ms = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]  # in seconds
    frame_indices = [int(20 * t) for t in times_ms] # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    mpjpe_specific_times = [[] for _ in times_ms]  # List to store MPJPE at specific times

    for idx, batch in enumerate(tqdm(subset_data_loader, desc="Sampling batches")):
        # print(f'### Sampling from batch {idx}/{total_batches}')
        input_motions, model_kwargs = batch
        input_motions = input_motions.to(dist_util.dev())
        # print(model_kwargs['y']['text'])
        # raise SystemExit

        ### For Motion Editing
        # model_kwargs['y']['inpainted_motion'] = input_motions # [bs, njoints, 1, seqlen]
        # model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
        #                                                         device=input_motions.device)  # True means use gt motion
        # for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
        #     model_kwargs['y']['inpainting_mask'][i, :, :, 50:] = False  # do inpainting in those frames
        # ################################

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        model_kwargs['y']['motion_embed'] = input_motions
        model_kwargs['y']['motion_embed_mask'] = torch.ones_like(input_motions, dtype=torch.bool, device=input_motions.device)
        model_kwargs['y']['motion_embed_mask'][:, :, :, start_idx:] = False
        sample_fn = diffusion.p_sample_loop

        for rep_i in range(args.num_repetitions):
            # print(f'### Sampling [repetitions #{rep_i}]')
            if args.lv:
                sample, log_variance = sample_fn(
                    model,
                    (args.batch_size, model.njoints, model.nfeats, max_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=False,
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
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )            

            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21 # sample.shape = [bs, 263, 1, 196]
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float() # [bs, 1, 196, 263]
                sample = recover_from_ric(sample, n_joints) # [bs, 1, 196, 22, 3]
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) # [bs, 22, 3, 196]
                if args.lv:
                    log_variance = data.dataset.t2m_dataset.inv_transform(log_variance.cpu().permute(0, 2, 3, 1)).float() # [bs, 1, 196, 263]
                    log_variance = log_variance[..., 4:(n_joints - 1) * 3 + 4] # [bs, 1, 196, 63]
                    log_variance = log_variance.view(log_variance.shape[:-1] + (-1, 3)) # [bs, 1, 196, 21, 3]
                    # log_variance = recover_from_ric(log_variance, n_joints)
                    log_variance = log_variance.view(-1, *log_variance.shape[2:]).permute(0, 2, 3, 1) # [bs, 21, 3, 196]

                input_motions_reshaped = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
                input_motions_reshaped = recover_from_ric(input_motions_reshaped, n_joints)
                input_motions_reshaped = input_motions_reshaped.view(-1, *input_motions_reshaped.shape[2:]).permute(0, 2, 3, 1)

            # Applying rot2xyz transformation
            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
            sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False) # [10, 22, 3, 196]

            if args.lv:
                log_variance = model.rot2xyz(x=log_variance, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                            jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                            get_rotations_back=False)
            
            input_motions_reshaped = model.rot2xyz(x=input_motions_reshaped, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                    jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                    get_rotations_back=False)
            
            valid_frame_mask = torch.zeros_like(input_motions_reshaped, dtype=torch.bool)

            for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
                valid_frame_mask[i, :, :, :length] = 1 # [64, 22, 3, 196]
                sample[i, :, :, length:] = 0
                input_motions_reshaped[i, :, :, length:] = 0

            # Compute MPJPE
            B, nb_joints, _, nb_frames = input_motions_reshaped.shape
            target_xyz_reshaped = input_motions_reshaped.permute(0, 3, 1, 2).reshape(args.batch_size, -1, 3) # torch.size([bs, 196*22, 3])
            pred_xyz_reshaped = sample.permute(0, 3, 1, 2).reshape(args.batch_size, -1, 3) # torch.size([bs, 196*22, 3])
            per_joint_errors = torch.norm(target_xyz_reshaped - pred_xyz_reshaped, 2, 2) # torch.Size([bs, 196*22])

            valid_frame_mask_reshaped = valid_frame_mask.reshape(args.batch_size, -1, 3).any(dim=2) # torch.Size([bs, 196*22])

            errors_reshaped = per_joint_errors.mean(dim=0) # Mean over batch #torch.Size([196*22])
            overtime_3d_err = errors_reshaped.view(-1, nb_joints).mean(dim=1) # torch.Size([196])

            # Compute MPJPE at specific time frames
            for t_idx, frame_idx in enumerate(frame_indices):
                valid_mask_at_frame = valid_frame_mask[:, :, :, frame_idx+10].any(dim=1).any(dim=0) # +10 is added because there is variability in the lengths of sequences within the same batch, so we prefer not to consider the last 10 frames to avoid this irregularity
                if valid_mask_at_frame.any().item():  
                    mpjpe_at_time = overtime_3d_err[frame_idx]
                    mpjpe_specific_times[t_idx].append(mpjpe_at_time.item())
                    adju_time = times_ms[t_idx] - start_time_ms
                    if times_ms[t_idx] <= start_time_ms:
                        continue
                    else:
                        print(f'Batch {idx} - Repetition {rep_i} - Time {adju_time}s - MPJPE: {mpjpe_at_time*1000:.4f}')

            if args.unconstrained:
                all_text += ['unconstrained'] * args.batch_size
            else:
                text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                all_text += model_kwargs['y'][text_key]

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
            if args.lv:
                all_variances.append(log_variance.cpu().numpy())

        # print(f"created {len(all_motions) * args.batch_size} samples")

    for time_ms, errors_at_time in zip(times_ms, mpjpe_specific_times):
        if time_ms <= start_time_ms:
            continue
        if errors_at_time:
            avg_error = sum(errors_at_time) / len(errors_at_time)
            adj_time = time_ms - start_time_ms
            print(f'---> MPJPE at {adj_time} s = {avg_error*1000:.4f}')


if __name__ == "__main__":
    main()
