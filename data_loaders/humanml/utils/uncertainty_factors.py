import torch

def compute_mean_absolute_change(means):
    """
    Compute the mean absolute change across successive denoising steps.
    
    Args:
        means (torch.Tensor): The mean values of shape [10, bs, 21, 196].
        
    Returns:
        torch.Tensor: The mean absolute change, shape [bs, 21, 196].
    """
    # Compute the difference between successive denoising steps
    changes = torch.abs(means[1:] - means[:-1])  # Shape: [9, bs, 21, 196]
    
    # Average the absolute changes across time steps
    mean_absolute_change = changes.mean(dim=0)  # Shape: [bs, 21, 196]
    return mean_absolute_change

def compute_max_change(means):
    """
    Compute the maximum change across successive denoising steps.
    
    Args:
        means (torch.Tensor): The mean values of shape [10, bs, 21, 196].
        
    Returns:
        torch.Tensor: The maximum change, shape [bs, 21, 196].
    """
    # Compute the difference between successive denoising steps
    changes = torch.abs(means[1:] - means[:-1])  # Shape: [9, bs, 21, 196]
    
    # Compute the maximum change across denoising steps
    max_change = changes.max(dim=0)[0]  # Shape: [bs, 21, 196]
    return max_change

def compute_cumulative_variance(means):
    """
    Compute the cumulative variance across denoising steps.
    
    Args:
        means (torch.Tensor): The mean values of shape [10, bs, 21, 196].
        
    Returns:
        torch.Tensor: The cumulative variance, shape [bs, 21, 196].
    """
    # Compute the variance across the last denoising steps
    cumulative_variance = torch.var(means, dim=0)  # Shape: [bs, 21, 196]
    return cumulative_variance

def compute_fluctuation_duration(means, threshold=0.05):
    """
    Compute the duration of fluctuations based on a threshold.
    
    Args:
        means (torch.Tensor): The mean values of shape [10, bs, 21, 196].
        threshold (float): The threshold value for significant fluctuations.
        
    Returns:
        torch.Tensor: The duration of significant fluctuations, shape [bs, 21, 196].
    """
    # Compute the difference between successive denoising steps
    changes = torch.abs(means[1:] - means[:-1])  # Shape: [9, bs, 21, 196]
    
    # Find where changes exceed a threshold
    fluctuation_mask = changes > threshold
    
    # Count the number of steps where fluctuations exceed the threshold
    fluctuation_duration = fluctuation_mask.sum(dim=0)  # Shape: [bs, 21, 196]
    return fluctuation_duration

def compute_relative_fluctuation(means):
    """
    Compute the relative fluctuation (percentage change) across denoising steps.
    
    Args:
        means (torch.Tensor): The mean values of shape [10, bs, 21, 196].
        
    Returns:
        torch.Tensor: The relative fluctuation, shape [bs, 21, 196].
    """
    # Compute the difference between successive denoising steps
    changes = torch.abs(means[1:] - means[:-1])  # Shape: [9, bs, 21, 196]
    
    # Normalize the change by the previous step to get relative change
    relative_fluctuation = changes / torch.abs(means[:-1] + 1e-6)  # Avoid division by zero
    
    # Compute the average relative fluctuation
    mean_relative_fluctuation = relative_fluctuation.mean(dim=0)  # Shape: [bs, 21, 196]
    return mean_relative_fluctuation
