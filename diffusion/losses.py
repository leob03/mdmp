# This code is initially based on https://github.com/openai/guided-diffusion and AUSE was added.
"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th

def calculate_ause(per_joint_errors, uncertainty_factor, lengths, n_bins=10):
    """
    Compute the Area Under Sparsification Error (AUSE) to assess the quality of the uncertainty factors.

    :param per_joint_errors: Tensor of per-joint errors. Shape: [bs, 196*22]
    :param uncertainty_factor: Tensor of uncertainty factors. Shape: # [bs, 196, 22]
    :param lengths: tensor of sequence actual length for each batch. Shape: [bs]
    :param n_bins: Number of bins to use for the AUSE calculation.
    """
    B, nb_frames, nb_joints = uncertainty_factor.shape  # B=bs, nb_frames=196, nb_joints=22

    per_joint_errors = per_joint_errors.view(B, nb_frames, nb_joints)  # Shape: [bs, 196, 22]

    # Find the shortest length in the batch
    shortest_length = lengths.min().item()

    # Truncate both tensors to the shortest length
    per_joint_errors = per_joint_errors[:, :shortest_length, :] # Shape: [bs, shortest_length, 21]
    uncertainty_factor = uncertainty_factor[:, :shortest_length, :] # Shape: [bs, shortest_length, 21]

    # Flatten the tensors to shape [bs * shortest_length * 21]
    per_joint_errors = per_joint_errors.reshape(-1)  # Shape: [bs * shortest_length * 21]
    uncertainty_factor = uncertainty_factor.reshape(-1)  # Shape: [bs * shortest_length * 21]

    # Sort by uncertainty
    sorted_indices_by_uncertainty = uncertainty_factor.argsort()
    sorted_errors_by_uncertainty = per_joint_errors[sorted_indices_by_uncertainty]  # Sorted per-joint errors by uncertainty
    sorted_indices_by_error = per_joint_errors.argsort()
    sorted_errors_by_error = per_joint_errors[sorted_indices_by_error]  # Sorted per-joint errors by error  

    # Calculate the total error (mean of all per-joint errors)
    total_error = sorted_errors_by_uncertainty.mean().item()

    # Compute sparsification error at different levels
    sparsification_errors = []
    sparsification_levels = []
    sparsification_errors_by_error = []
    for i in range(1, n_bins+1):
        # Sparsification level: keep (1 - i/n_bins) fraction of the data
        threshold_index = int((1 - i/n_bins) * len(sorted_errors_by_uncertainty))
        if threshold_index == 0:
            continue
        
        sparsified_errors = sorted_errors_by_uncertainty[:threshold_index]  # Shape: [threshold_index]
        sparsified_error = sparsified_errors.mean().item()
        # uncertainty_ration = (total_error - sparsified_error) / sparsified_error

        sparsified_errors_by_error = sorted_errors_by_error[:threshold_index]  # Shape: [threshold_index]
        sparsified_error_by_error = sparsified_errors_by_error.mean().item()
        # error_ratio = (total_error - sparsified_error_by_error) / sparsified_error_by_error

        sparsification_errors.append(sparsified_error)
        sparsification_errors_by_error.append(sparsified_error_by_error)

        # sparsification_errors.append(uncertainty_ration)
        # sparsification_errors_by_error.append(error_ratio)

        sparsification_levels.append(i/n_bins)

    return sparsification_errors, sparsification_errors_by_error, sparsification_levels
    

def calculate_ause2(per_joint_errors, uncertainty_factor, lengths, output_path, n_bins=10):
    """
    Compute the Area Under Sparsification Error (AUSE) to assess the quality of the uncertainty factors.

    :param per_joint_errors: Tensor of per-joint errors. Shape: [bs, 196*22]
    :param uncertainty_factor: Tensor of uncertainty factors. Shape: [bs, 21, 3, 196]
    :param lengths: tensor of sequence actual length for each batch. Shape: [bs]
    :param n_bins: Number of bins to use for the AUSE calculation.
    """
    B, nb_joints, _, nb_frames = uncertainty_factor.shape  # B=bs, nb_joints=21, nb_frames=196
    
    # Reduce uncertainty_factor over the 3D space dimension (x, y, z)
    uncertainty_factor = uncertainty_factor.mean(dim=2)  # New shape: [bs, 21, 196]
    uncertainty_factor = uncertainty_factor.permute(0, 2, 1)  # New shape: [bs, 196, 21]

    per_joint_errors = per_joint_errors.view(B, nb_frames, nb_joints+1)  # Shape: [bs, 196, 22]
    # Exclude the root joint (joint 0)
    per_joint_errors = per_joint_errors[:, :, 1:]  # Shape: [bs, 196, 21]

    # Find the shortest length in the batch
    shortest_length = lengths.min().item()

    # Truncate both tensors to the shortest length
    per_joint_errors = per_joint_errors[:, :shortest_length, :] # Shape: [bs, shortest_length, 21]
    uncertainty_factor = uncertainty_factor[:, :shortest_length, :] # Shape: [bs, shortest_length, 21]

    # Flatten the tensors to shape [bs * shortest_length * 21]
    per_joint_errors = per_joint_errors.reshape(-1)  # Shape: [bs * shortest_length * 21]
    uncertainty_factor = uncertainty_factor.reshape(-1)  # Shape: [bs * shortest_length * 21]

    # Sort by uncertainty
    sorted_indices_by_uncertainty = uncertainty_factor.argsort()
    sorted_errors_by_uncertainty = per_joint_errors[sorted_indices_by_uncertainty]  # Sorted per-joint errors by uncertainty
    sorted_indices_by_error = per_joint_errors.argsort()
    sorted_errors_by_error = uncertainty_factor[sorted_indices_by_error]  # Sorted per-joint errors by error   

    # Compute sparsification error at different levels
    sparsification_errors = []
    sparsification_levels = []
    for i in range(1, n_bins+1):
        # Sparsification level: keep (1 - i/n_bins) fraction of the data
        threshold_index = int((1 - i/n_bins) * len(sorted_errors_by_uncertainty))
        if threshold_index == 0:
            continue
        
        subset_u = sorted_errors_by_uncertainty[:threshold_index]  # Shape: [threshold_index]
        subset_u = subset_u.mean().item()
        subset_e = sorted_errors_by_error[:threshold_index]  # Shape: [threshold_index]
        subset_e = subset_e.mean().item()

        # Calculate the sparsification error
        sparsification_error = subset_u - subset_e
        sparsification_errors.append(sparsification_error)
        sparsification_levels.append(i/n_bins)

    # Plot the sparsification error curve
    return sparsification_errors, sparsification_levels

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
