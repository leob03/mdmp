# This code is initially based on https://github.com/openai/guided-diffusion and AUSE was added.
"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th

# def calculate_ause(per_joint_errors, uncertainty_factor, valid_frame_mask, n_bins=10):
#     """
#     Compute the Area Under Sparsification Error (AUSE) to assess the quality of the uncertainty factors.
#     """
#     # Flatten the errors and variances
#     per_joint_errors = per_joint_errors.flatten()
#     uncertainty_factor = uncertainty_factor.flatten()
#     valid_frame_mask = valid_frame_mask.flatten()

#     # Filter valid frames
#     per_joint_errors = per_joint_errors[valid_frame_mask]
#     uncertainty_factor = uncertainty_factor[valid_frame_mask]

#     # Sort errors and variances by increasing variance
#     sorted_indices = uncertainty_factor.argsort()
#     sorted_errors = per_joint_errors[sorted_indices]
#     sorted_variances = uncertainty_factor[sorted_indices]

#     # Calculate the total error
#     total_error = sorted_errors.mean()

#     # Compute sparsification error at different levels
#     sparsification_errors = []
#     for i in range(1, n_bins + 1):
#         # Sparsification level: keep (1 - i/n_bins) fraction of the data
#         threshold_index = int((1 - i/n_bins) * len(sorted_errors))
#         sparsified_errors = sorted_errors[:threshold_index]
#         sparsified_error = sparsified_errors.mean()

#         # Calculate the sparsification error
#         sparsification_error = sparsified_error - total_error
#         sparsification_errors.append(sparsification_error)

#     # Calculate AUSE as the area under the sparsification error curve
#     ause = th.trapz(th.tensor(sparsification_errors), dx=1.0/n_bins)

#     return ause.item()

def calculate_ause(per_joint_errors, uncertainty_factor, valid_frame_mask, n_bins=10):
    """
    Compute the Area Under Sparsification Error (AUSE) to assess the quality of the uncertainty factors.

    :param per_joint_errors: Tensor of per-joint errors. Shape: [bs, 196*22]
    :param uncertainty_factor: Tensor of uncertainty factors. Shape: [bs, 21, 3, 196]
    :param valid_frame_mask: Tensor of valid frame masks. Shape: [bs, 196*22]
    :param n_bins: Number of bins to use for the AUSE calculation.
    """
    B, nb_joints, _, nb_frames = uncertainty_factor.shape  # B=bs, nb_joints=21, nb_frames=196
    
    # Reduce uncertainty_factor over the 3D space dimension (x, y, z)
    uncertainty_factor = uncertainty_factor.mean(dim=2)  # New shape: [bs, 21, 196]
    uncertainty_factor = uncertainty_factor.permute(0, 2, 1).reshape(-1, 196 * 21)  # Shape: [bs, 196*21]

    per_joint_errors = per_joint_errors.view(B, nb_frames, nb_joints+1)  # Shape: [bs, 196, 22]
    # Exclude the root joint (joint 0)
    per_joint_errors = per_joint_errors[:, :, 1:]  # Shape: [bs, 196, 21]
    per_joint_errors = per_joint_errors.reshape(B, -1)  # Shape: [bs, 196*21]

    # Flatten the errors and uncertainty_factor -> bin per batch+frame+joint
    per_joint_errors = per_joint_errors.flatten()  # Shape: [bs * 196 * 21]
    uncertainty_factor = uncertainty_factor.flatten()  # Shape: [bs * 196 * 21]
    # valid_frame_mask = valid_frame_mask[:, :196 * 21].flatten()  # Shape: [bs * 196 * 21]

    # Filter valid frames
    # per_joint_errors = per_joint_errors[valid_frame_mask]
    # uncertainty_factor = uncertainty_factor[valid_frame_mask]

    # Step 6: Sort errors and uncertainty_factor by increasing uncertainty_factor
    sorted_indices = uncertainty_factor.argsort()
    sorted_errors = per_joint_errors[sorted_indices]
    sorted_uncertainty = uncertainty_factor[sorted_indices]

    # Step 7: Calculate the total error
    total_error = sorted_errors.mean()

    # Step 8: Compute sparsification error at different levels
    sparsification_errors = []
    for i in range(1, n_bins + 1):
        # Sparsification level: keep (1 - i/n_bins) fraction of the data
        threshold_index = int((1 - i/n_bins) * len(sorted_errors))
        sparsified_errors = sorted_errors[:threshold_index]
        sparsified_error = sparsified_errors.mean()

        # Calculate the sparsification error
        sparsification_error = sparsified_error - total_error
        sparsification_errors.append(sparsification_error)

    # Step 9: Calculate AUSE as the area under the sparsification error curve
    ause = th.trapz(th.tensor(sparsification_errors), dx=1.0/n_bins)

    return ause.item()


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
