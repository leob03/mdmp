from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
# from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdmp_loader  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel

import logging

# Configure logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_mpjpe(eval_wrapper, motion_loaders, file, learning_var, start_idx, get_xyz):
    mpjpe_dict = OrderedDict({})
    print('========== Evaluating MPJPE ==========')

    times_ms = [400, 1000, 1500, 2000]  # in milliseconds
    frame_indices = [int(20 * (t / 1000.0)) - start_idx for t in times_ms]  # convert ms to frame index, adjust for start_idx

    for motion_loader_name, motion_loader in motion_loaders.items():
        mean_errors = []
        mpjpe_specific_times = [[] for _ in times_ms]  # List to store MPJPE at specific times
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                if motion_loader_name == 'ground truth':
                    continue
                else:
                    _, _, _, _, input_motion, motion, _, _, _ = batch
                
                logging.debug('batch_ix: %d', idx)
                logging.debug('motion shape: %s', motion.shape)
                logging.debug('motion: %s', motion)
                input_motion = input_motion.unsqueeze(1).float() # torch.Size([32, 1, 196, 263])
                motion = motion.unsqueeze(1).float() # torch.Size([32, 1, 196, 263])
                # print(f"motion:" , motion)

                if torch.isnan(input_motion).any() or torch.isinf(input_motion).any():
                    print(f"NaN or Inf found in input_motion at batch {idx}")
                if torch.isnan(motion).any() or torch.isinf(motion).any():
                    print(f"NaN or Inf found in motion at batch {idx}")
                
                n_joints = 22 if input_motion.shape[3] == 263 else 21
                input_motion = recover_from_ric(input_motion, n_joints)
                input_motion = input_motion.view(-1, *input_motion.shape[2:]).permute(0, 2, 3, 1)

                motion = recover_from_ric(motion, n_joints)
                motion = motion.view(-1, *motion.shape[2:]).permute(0, 2, 3, 1)

                if torch.isnan(input_motion).any() or torch.isinf(input_motion).any():
                    print(f"NaN or Inf found in input_motion at batch {idx} after recover_from_ric")
                if torch.isnan(motion).any() or torch.isinf(motion).any():
                    print(f"NaN or Inf found in motion at batch {idx} after recover_from_ric")

                # Convert to XYZ format
                target_xyz = get_xyz(input_motion)  # (B, nb_joints, 3, nb_frames)
                pred_xyz = get_xyz(motion)

                if torch.isnan(target_xyz).any() or torch.isinf(target_xyz).any():
                    print(f"NaN or Inf found in target_xyz at batch {idx}")

                if torch.isnan(pred_xyz).any() or torch.isinf(pred_xyz).any():
                    print(f"NaN or Inf found in pred_xyz at batch {idx}")

                B, nb_joints, _, nb_frames = target_xyz.shape

                # mask = torch.ones_like(pred_xyz, dtype=torch.bool, device=target_xyz.device)
                # mask[:, :, :, :start_idx] = False

                # # Apply mask using element-wise multiplication
                # masked_target_xyz = target_xyz * mask.float()
                # masked_pred_xyz = pred_xyz * mask.float()

                target_xyz = target_xyz[:, :, :,start_idx:]  # (B, nb_joints, 3, nb_frames - start_idx)
                pred_xyz = pred_xyz[:, :, :,start_idx:]
                
                target_xyz_reshaped = target_xyz.permute(0, 3, 1, 2).reshape(32, -1, 3)
                pred_xyz_reshaped = pred_xyz.permute(0, 3, 1, 2).reshape(32, -1, 3)

                # Compute the per-joint position error for each frame
                per_joint_errors = torch.norm(target_xyz_reshaped - pred_xyz_reshaped, 2, 2) # torch.Size([32, 196*22])                

                if torch.isnan(per_joint_errors).any() or torch.isinf(per_joint_errors).any():
                    print(f"NaN or Inf found in per_joint_errors at batch {idx}")

                errors_reshaped = per_joint_errors.mean(dim=0)  # Mean over batch #torch.Size([196*22])

                if torch.isnan(errors_reshaped).any() or torch.isinf(errors_reshaped).any():
                    print(f"NaN or Inf found in errors_reshaped at batch {idx}")

                # Compute the mean error across all joints and frames for overall MPJPE
                overtime_3d_err = errors_reshaped.reshape(-1, nb_joints).mean(dim=1)  # torch.Size([196])

                if torch.isnan(overtime_3d_err).any() or torch.isinf(overtime_3d_err).any():
                    print(f"NaN or Inf found in overtime_3d_err at batch {idx}")

                mean_3d_err = overtime_3d_err.mean() # torch.Size([])

                if torch.isnan(mean_3d_err).any() or torch.isinf(mean_3d_err).any():
                    print(f"NaN or Inf found in mean_3d_err at batch {idx}")

                mean_errors.append(mean_3d_err.item())

                # Compute MPJPE at specific time frames
                for idx, frame_idx in enumerate(frame_indices):
                    if frame_idx < nb_frames - start_idx:
                        mpjpe_at_time = overtime_3d_err[frame_idx]
                        mpjpe_specific_times[idx].append(mpjpe_at_time.item())

        mpjpe_dict[motion_loader_name] = sum(mean_errors) / len(mean_errors) if mean_errors else float('inf')
        print(f'---> [{motion_loader_name}]: Overall MPJPE = {mpjpe_dict[motion_loader_name]:.4f}')
        for time_ms, errors_at_time in zip(times_ms, mpjpe_specific_times):
            if errors_at_time:
                avg_error = sum(errors_at_time) / len(errors_at_time)
                print(f'---> [{motion_loader_name}]: MPJPE at {time_ms} ms = {avg_error:.4f}')
                print(f'---> [{motion_loader_name}]: MPJPE at {time_ms} ms = {avg_error:.4f}', file=file, flush=True)

    return mpjpe_dict

# def evaluate_ngll(eval_wrapper, motion_loaders, file, learning_var, start_idx, get_xyz):
#     mjpje_dict = OrderedDict({})
#     print('========== Evaluating NGLL ==========')

#     for motion_loader_name, motion_loader in motion_loaders.items():
#         mean_errors = []
#         with torch.no_grad():
#             for idx, batch in enumerate(motion_loader):
#                 if learning_var:
#                     _, _, _, _, input_motion, motion, log_variance, _, _ = batch -------> here
#                 else:
#                     _, _, _, _, input_motion, motion, _, _ = batch
                
#                 B, nb_joints, _, nb_frames = input_motion.shape

#                 # Convert to XYZ format
#                 target_xyz = get_xyz(input_motion)  # (B, nb_joints, 3, nb_frames)
#                 pred_xyz = get_xyz(motion)

#                 mask = torch.ones_like(target_xyz, dtype=torch.bool, device=target_xyz.device)
#                 mask[:, :, :, :start_idx] = False
                
#                 target_xyz_reshaped = target_xyz.permute(0, 3, 1, 2).reshape(-1, nb_joints, 3)
#                 pred_xyz_reshaped = pred_xyz.permute(0, 3, 1, 2).reshape(-1, nb_joints, 3)
#                 mask_reshaped = mask.permute(0, 3, 1, 2).reshape(-1, nb_joints, 3)

#                 # Apply the mask to filter out the relevant values
#                 masked_target_xyz = target_xyz_reshaped[mask_reshaped]
#                 masked_pred_xyz = pred_xyz_reshaped[mask_reshaped]

#                 # Compute the per-joint position error for each frame
#                 per_joint_errors = torch.norm(masked_target_xyz - masked_pred_xyz, dim=1)

#                 # Compute the mean error across all joints and frames
#                 mean_3d_err = torch.mean(per_joint_errors)
#                 mean_errors.append(mean_3d_err.item())

#         mjpje_dict[motion_loader_name] = sum(mean_errors) / len(mean_errors) if mean_errors else float('inf')

#         print(f'---> [{motion_loader_name}]: MJPJE = {mjpje_dict[motion_loader_name]:.4f}')
#         print(f'---> [{motion_loader_name}]: MJPJE = {mjpje_dict[motion_loader_name]:.4f}', file=file, flush=True)

#     return mjpje_dict


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                # print(f"Batch {idx}: {batch}")  # Debug print
                if motion_loader_name == 'ground truth':
                    # print('here for gt')
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                else:
                    # print('here for mdmp_motion_loader')
                    word_embeddings, pos_one_hots, _, sent_lens, _, motions, _, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, learning_var, start_idx, get_xyz, run_mm=False):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'MPJPE': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            # print(f'Time: {datetime.now()}')
            # print(f'Time: {datetime.now()}', file=f, flush=True)
            # fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            # print(f'Time: {datetime.now()}')
            # print(f'Time: {datetime.now()}', file=f, flush=True)
            # div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mpjpe_dict = evaluate_mpjpe(eval_wrapper, motion_loaders, f, learning_var, start_idx, get_xyz)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            # for key, item in fid_score_dict.items():
            #     if key not in all_metrics['FID']:
            #         all_metrics['FID'][key] = [item]
            #     else:
            #         all_metrics['FID'][key] += [item]

            # for key, item in div_score_dict.items():
            #     if key not in all_metrics['Diversity']:
            #         all_metrics['Diversity'][key] = [item]
            #     else:
            #         all_metrics['Diversity'][key] += [item]
            
            for key, item in mpjpe_dict.items():
                if key not in all_metrics['MPJPE']:
                    all_metrics['MPJPE'][key] = [item]
                else:
                    all_metrics['MPJPE'][key] += [item]
                    
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    start_idx = args.emb_motion_len
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_{}_{}'.format(name, niter))
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')

    print(f'Eval mode [{args.eval_mode}]')
    if args.eval_mode == 'debug':
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 1
        # replication_times = 5  # about 3 Hrs

    elif args.eval_mode == 'wo_mm':
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 20 # about 12 Hrs
    elif args.eval_mode == 'mm_short':
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt')
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval')
    num_actions = gen_loader.dataset.num_actions

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)
    learning_var = model.learning_var
    enc = model
    get_xyz = lambda sample : enc.rot2xyz(sample, mask=None, pose_rep='xyz', glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        'vald': lambda: get_mdmp_loader(
            model, diffusion, args.batch_size,
            gen_loader, mm_num_samples, mm_num_repeats, gt_loader.dataset.opt.max_motion_length, num_samples_limit, args.guidance_param
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, learning_var, start_idx, get_xyz, run_mm=run_mm)
