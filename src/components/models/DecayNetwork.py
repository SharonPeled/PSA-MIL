import torch
import torch.nn as nn
import numpy as np


class DecayNetwork(nn.Module):
    def __init__(self, decay_type, num_heads, decay_clip=None,
                 min_local_k=1, max_local_k=25, init_k=7):
        super(DecayNetwork, self).__init__()
        self.decay_type = decay_type
        self.decay_clip = decay_clip
        self.min_local_k = min_local_k
        self.max_local_k = max_local_k

        self.theta1 = compute_theta_from_local_k(decay_type=self.decay_type, decay_clip=self.decay_clip,
                                                 local_k=self.min_local_k)
        self.theta2 = compute_theta_from_local_k(decay_type=self.decay_type, decay_clip=self.decay_clip,
                                                 local_k=self.max_local_k)

        self.theta_min = self.theta1 if self.theta1 < self.theta2 else self.theta2
        self.theta_max = self.theta1 if self.theta1 > self.theta2 else self.theta2

        self.lambda_p = nn.Parameter(self.init_around_k(init_k, num_heads))

    def init_around_k(self, target_k, num_heads):
        init_low_bound = compute_theta_from_local_k(decay_type=self.decay_type, decay_clip=self.decay_clip,
                                                    local_k=target_k - 1)
        init_high_bound = compute_theta_from_local_k(decay_type=self.decay_type, decay_clip=self.decay_clip,
                                                     local_k=target_k + 1)
        theta_init_min = min(init_low_bound, init_high_bound)
        theta_init_max = max(init_low_bound, init_high_bound)
        theta_range = theta_init_max - theta_init_min
        theta_init_min += theta_range*0.25
        theta_init_max -= theta_range*0.25

        lambda_min = (theta_init_min - self.theta_min) / (self.theta_max - self.theta_min)
        lambda_max = (theta_init_max - self.theta_min) / (self.theta_max - self.theta_min)

        return torch.linspace(lambda_min, lambda_max, num_heads)

    def reparam(self, lambda_p):
        lambda_p = torch.clamp(lambda_p, min=0, max=1)
        theta = self.theta_min + lambda_p * (self.theta_max - self.theta_min)
        return theta

    def get_params(self, param_str):
        if param_str == 'rates':
            lambda_p = torch.clamp(self.lambda_p, min=0, max=1)
            return lambda_p
        elif param_str == 'thetas':
            return self.reparam(self.lambda_p)
        elif param_str == 'local_Ks':
            return torch.round(solve_for_local_k(self.decay_type, self.reparam(self.lambda_p), self.decay_clip))

    def forward(self, distance, decay_clip=None, head_ind=None):
        lambda_p = self.reparam(self.lambda_p)
        decays = self.spatial_decay(distance, lambda_p[head_ind])
        return decays

    def spatial_decay(self, distance, p):
        decay_type = self.decay_type
        if decay_type is None:
            return torch.zeros_like(distance)
        elif decay_type == 'Gaussian':
            return gaussian_decay(distance, p)
        elif decay_type == 'Exponential':
            return exponential_decay(distance, p)
        elif decay_type == 'InverseQuadratic':
            return inverse_quadratic_decay(distance, p)
        else:
            raise ValueError(f"Unknown DECAY_TYPE: {decay_type}")


def uniform_sample(a: float, b: float, shape: tuple):
    return (b - a) * torch.rand(shape) + a


def compute_theta_from_local_k(decay_type, local_k, decay_clip):
    """
    Computes the decay parameter theta (sigma, lambda, or p) given a decay type, local_k, and decay_clip.

    Parameters:
    - decay_type: str, type of decay function ('Gaussian', 'Exponential', 'InverseQuadratic')
    - local_k: float, the cutoff distance
    - decay_clip: float, the threshold value at local_k

    Returns:
    - theta: float, the computed decay parameter (sigma, lambda, or p)
    """

    decay_clip = torch.tensor(decay_clip, dtype=torch.float32)
    local_k = torch.tensor(local_k, dtype=torch.float32)

    if decay_type == 'Gaussian':
        # sigma = sqrt(local_k^2 / (-2 log(decay_clip)))
        theta = torch.sqrt(local_k ** 2 / (-2 * torch.log(decay_clip)))

    elif decay_type == 'Exponential':
        # lambda = -log(decay_clip) / local_k
        theta = -torch.log(decay_clip) / local_k

    elif decay_type == 'InverseQuadratic':
        # p = local_k / sqrt[(1 - decay_clip) / decay_clip]
        theta = local_k / torch.sqrt((1 - decay_clip) / decay_clip)

    else:
        raise ValueError("Invalid decay type. Choose from 'Gaussian', 'Exponential', or 'InverseQuadratic'.")

    return theta.item()  # Convert to a Python float


def solve_for_local_k(decay_type, param, decay_clip):
    decay_clip = torch.tensor(decay_clip, dtype=param.dtype, device=param.device)
    if decay_type == 'Gaussian':
        # Gaussian decay: f(d | sigma) = exp(- (d^2) / (2 * sigma^2))
        # Rearranging to solve for local_k: d<= sqrt[log(decay_clip) * 2 * sigma**2]
        local_k = torch.sqrt(-torch.log(decay_clip) * 2 * (param**2))
        return local_k

    elif decay_type == 'Exponential':
        # Exponential decay: f(d | lambda) = exp(-lambda * d)
        # Rearranging to solve for local_k: d <= -log(decay_clip) / lambda
        local_k = -torch.log(decay_clip) / param
        return local_k

    elif decay_type == 'InverseQuadratic':
        # Inverse quadratic decay: f(d | p) = 1 / (1 + (d / p)^2)
        # Rearranging to solve for local_k: d < = sqrt[ ((1-decay_clip)/decay_clip)* p**2 ]
        local_k = torch.sqrt(((1-decay_clip)/decay_clip) * (param**2))
        return local_k
    else:
        raise ValueError("Invalid decay type. Choose from 'Gaussian', 'Exponential', or 'InverseQuadratic'.")


def solve_for_theta(decay_type, target_decay, distance):
    """
    Solves for the decay parameter theta (sigma, lambda, or p) given a decay type and target decay value.

    Parameters:
    - decay_type: str, the type of decay function ('Gaussian', 'Exponential', 'InverseQuadratic')
    - target_decay: float, the desired decay value (default is 0.1)
    - distance: float, the distance at which the decay should reach the target value (default is 10)

    Returns:
    - theta: float, the computed decay parameter (sigma, lambda, or p)
    """

    if decay_type == 'Gaussian':
        # Gaussian decay: f(d | sigma) = exp(- (d^2) / (2 * sigma^2))
        # Rearranging to solve for sigma: sigma = sqrt(-d^2 / (2 * log(target_decay)))
        sigma = np.sqrt(-distance ** 2 / (2 * np.log(target_decay)))
        return sigma

    elif decay_type == 'Exponential':
        # Exponential decay: f(d | lambda) = exp(-lambda * d)
        # Rearranging to solve for lambda: lambda = -log(target_decay) / d
        lambda_param = -np.log(target_decay) / distance
        # lambda_param = 1 / lambda_param
        return lambda_param

    elif decay_type == 'InverseQuadratic':
        # Inverse quadratic decay: f(d | p) = 1 / (1 + (d / p)^2)
        # Rearranging to solve for p: p = d / sqrt((1 / target_decay) - 1)
        p = distance / np.sqrt((1 / target_decay) - 1)
        return p

    elif decay_type is None:
        return 1 / target_decay * distance  # just random fix
    else:
        raise ValueError("Invalid decay type. Choose from 'Gaussian', 'Exponential', or 'InverseQuadratic'.")


def gaussian_decay(distance, sigma):
    """
    Gaussian decay function.

    decay = exp(- (distance^2) / (2 * sigma^2))

    Parameters:
    - distance: [B, num_heads, N, N]
    - sigma: [B, num_heads]

    Returns:
    - decay: [B, num_heads, N, N]
    """
    # Expand sigma for broadcasting to [B, num_heads, 1, 1]
    # sigma_expanded = (1/sigma).unsqueeze(-1).unsqueeze(-1)  # [B, num_heads, 1, 1]
    sigma_expanded = sigma.unsqueeze(-1).unsqueeze(-1)  # [B, num_heads, 1, 1]

    # Compute decay
    decay = torch.exp(
        - (distance ** 2) / (2 * sigma_expanded ** 2))  # [B, num_heads, N, N]

    return decay


def exponential_decay(distance, lambda_sample):
    """
    Exponential decay function.

    decay = exp(-lambda_sample * distance)

    Parameters:
    - distance: [B, seq_len, seq_len]
    - lambda_sample: [B, num_heads]

    Returns:
    - decay: [B, num_heads, seq_len, seq_len]
    """
    # Expand lambda_sample for broadcasting to [B, num_heads, 1, 1]
    lambda_sample_expanded = lambda_sample.unsqueeze(-1).unsqueeze(-1)  # [B, num_heads, 1, 1]

    # Compute decay
    decay = torch.exp(-lambda_sample_expanded * distance)  # [B, num_heads, seq_len, seq_len]

    return decay


def inverse_quadratic_decay(distance, sampled_p):
    """
    Inverse quadratic decay function: 1 / (1 + (d / sampled_p)^2)

    Parameters:
    - distance: [B, num_heads, N, N]
    - sampled_p: [B, num_heads]

    Returns:
    - decay: [B, num_heads, N, N]
    """
    # Expand sampled_p for broadcasting to [B, num_heads, 1, 1]
    # p_expanded = (1/sampled_p).unsqueeze(-1).unsqueeze(-1)  # [B, num_heads, 1, 1]
    p_expanded = sampled_p.unsqueeze(-1).unsqueeze(-1)  # [B, num_heads, 1, 1]

    # Compute decay
    decay = 1 / (1 + (distance / p_expanded) ** 2)  # [B, num_heads, N, N]

    return decay


def inverse_gaussian_decay(distance, decay):
    """
    Compute sigma for a given distance and decay in Gaussian decay function.

    Parameters:
    - distance: scalar or tensor
    - decay: scalar or tensor (0 < decay <= 1)

    Returns:
    - sigma
    """
    sigma = distance / torch.sqrt(2 * -torch.log(decay))
    return sigma


def inverse_exponential_decay(distance, decay):
    """
    Compute lambda for a given distance and decay in Exponential decay function.

    Parameters:
    - distance: scalar or tensor
    - decay: scalar or tensor (0 < decay <= 1)

    Returns:
    - lambda_sample
    """
    lambda_sample = -torch.log(decay) / distance
    return lambda_sample


def inverse_inverse_quadratic_decay(distance, decay):
    """
    Compute sampled_p for a given distance and decay in Inverse Quadratic decay function.

    Parameters:
    - distance: scalar or tensor
    - decay: scalar or tensor (0 < decay <= 1)

    Returns:
    - sampled_p
    """
    denom = torch.sqrt(1 / decay - 1)
    sampled_p = distance / denom
    return sampled_p

