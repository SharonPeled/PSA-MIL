import torch
import torch.nn as nn
from src.components.models.ABMILClassifier import ABMILClassifier
import numpy as np


class DecayNetwork(nn.Module):
    def __init__(self, decay_type, embed_dim, num_heads, slide_repr_condition=None, decay_clip=None,
                 min_local_k=1, max_local_k=25):
        # bigger theta - smaller decay (less local)
        # small theta - big decay (local)
        super(DecayNetwork, self).__init__()
        if decay_type is None:
            # TODO: fix bandaid for local and rpe
            self.lambda_p = nn.Parameter(uniform_sample(0.1, 0.6, num_heads))
            return
        self.decay_type = decay_type
        self.decay_clip = decay_clip
        self.min_local_k = min_local_k
        self.max_local_k = max_local_k
        self.slide_repr_condition = slide_repr_condition
        self.magnitude = nn.Parameter(torch.ones(1))

        self.theta1 = compute_theta_from_local_k(decay_type=self.decay_type, decay_clip=self.decay_clip,
                                                 local_k=self.min_local_k)
        self.theta2 = compute_theta_from_local_k(decay_type=self.decay_type, decay_clip=self.decay_clip,
                                                 local_k=self.max_local_k)
        self.theta_min = self.theta1 if self.theta1 < self.theta2 else self.theta2
        self.theta_max = self.theta1 if self.theta1 > self.theta2 else self.theta2
        # self.lambda_p = nn.Parameter(self.uniform_distribution(num_heads))
        target_init_k = 7
        self.lambda_p = nn.Parameter(self.init_around_k(target_init_k, num_heads))

    def init_around_k(self, target_k, num_heads):
        # init value between 0 to 1
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

    def uniform_distribution(self, num_heads, start=0, end=1):
        # Generate n equally spaced values between 0 and 1
        values = torch.linspace(start, end, num_heads + 2)[1:-1]  # Exclude 0 and 1
        return values

    def init_theta(self, num_heads, theta_min, theta_max):
        # return torch.randn(num_heads)
        theta_target = solve_for_theta(self.decay_type, distance=10, target_decay=0.1)
        a = (theta_target - theta_min) / (theta_max - theta_min)
        theta_target = solve_for_theta(self.decay_type, distance=20, target_decay=0.1)
        b = (theta_target - theta_min) / (theta_max - theta_min)
        # print(a,b, theta_min, theta_max)
        return torch.tensor([a, b][:num_heads])

    def reparam(self, lambda_p):
        # if self.decay_type == 'Exponential':
        #     lambda_p = 1 / lambda_p
        lambda_p = torch.clamp(lambda_p, min=0, max=1)
        theta = self.theta_min + lambda_p * (self.theta_max - self.theta_min)
        # print(theta, lambda_p)
        return theta

    def get_slide_repr(self, tile_embeddings, coords):
        if self.slide_repr_condition == 'Mean-Pool':
            slide_repr = tile_embeddings.mean(dim=1)  # [B, dim]
            return slide_repr
        if self.slide_repr_condition == 'MoransI':
            return morans_I(tile_embeddings, coords)
        if self.slide_repr_condition == 'ABMIL':
            slide_reprs = []
            for i in range(len(tile_embeddings)):
                slide_reprs.append(self.lambda_p_net_abmil.single_bag_forward(tile_embeddings[i])[1])
            return torch.stack(slide_reprs, dim=0)
        return None

    def get_params(self, param_str):
        if param_str == 'rates':
            lambda_p = torch.clamp(self.lambda_p, min=0, max=1)
            return lambda_p
        elif param_str == 'thetas':
            return self.reparam(self.lambda_p)
        elif param_str == 'local_Ks':
            return torch.round(solve_for_local_k(self.decay_type, self.reparam(self.lambda_p), self.decay_clip))

    def forward(self, distance, tile_embeddings=None, coords=None, decay_clip=None, head_ind=None):
        # print(self.lambda_p)
        if self.slide_repr_condition is None:
            # lambda_p = self.lambda_p
            # lambda_p = self.ln(self.lambda_p)
            # lambda_p = torch.clamp(lambda_p, min=1e-6)
            lambda_p = self.reparam(self.lambda_p)
            # print(lambda_p)
        elif self.slide_repr_condition == 'MAG':
            magnitude = torch.clamp(self.magnitude, min=1e-6)
            lambda_p = self.lambda_p.softmax(dim=-1)
            lambda_p = magnitude * lambda_p
        else:
            slide_repr = self.get_slide_repr(tile_embeddings, coords)
            lambda_p = self.lambda_p_net(slide_repr)
            lambda_p = torch.clamp(lambda_p, min=1e-6)
            lambda_p = self.ln(lambda_p)
            lambda_p = torch.clamp(lambda_p, min=1e-6)

        if head_ind is not None:
            decays = self.spatial_decay(distance, lambda_p[head_ind])
            return decays, lambda_p[head_ind]

        decays = self.spatial_decay(distance, lambda_p)

        if decay_clip is not None:
            decays = torch.where(decays < decay_clip, torch.zeros_like(decays), decays)
        return decays.float(), lambda_p

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





def morans_I(tile_embeddings, coords):
    B, N, D = tile_embeddings.shape  # Batch size, number of tiles, feature dimension

    # coords is B x N x 2 (assuming last two dimensions are row and col)
    coords = coords.unsqueeze(2)  # B x N x 1 x 2
    distance = ((coords - coords.transpose(1, 2)) ** 2).sum(dim=-1)  # Broadcasting, B x N x N
    distance = distance.sqrt()

    # Step 2: Compute the weight matrix W as 1/d, with 0s on the diagonal
    W = 1 / (distance + 1e-6)  # [B, N, N]

    # Set diagonal of each matrix in the batch to 0
    eye_mask = torch.eye(N, device=coords.device).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
    W = W * (1 - eye_mask)  # Set diagonal to 0

    # Step 3: Compute the mean embedding for each feature in the batch
    mean_tile_embeddings = torch.mean(tile_embeddings, dim=1, keepdim=True)  # [B, 1, D]

    # Step 4: Compute z_i - z_mean for each tile embedding and each feature
    z_diff = tile_embeddings - mean_tile_embeddings  # [B, N, D]

    # Step 5: Compute Moran's I for each batch and each feature incrementally
    # We will calculate the numerator in a loop to reduce memory consumption
    numerator = torch.zeros((B, D), device=tile_embeddings.device)  # Initialize numerator [B, D]

    # Incrementally compute the numerator to reduce memory footprint
    for i in range(N):
        z_diff_i = z_diff[:, i, :].unsqueeze(1)  # [B, 1, D]
        W_i = W[:, i, :].unsqueeze(-1)  # [B, N, 1]
        z_pairwise_prod_i = z_diff_i * z_diff  # [B, N, D]
        numerator += torch.sum(W_i * z_pairwise_prod_i, dim=1)  # Sum over tiles for current row i

    # Denominator of Moran's I: sum((z_i - mean_z)^2) for each feature
    denominator = torch.sum(z_diff ** 2, dim=1)  # [B, D]

    # Normalizing term: N / sum(W_ij) for each batch
    W_sum = torch.sum(W, dim=(1, 2))  # [B]

    # Moran's I computation for each batch and each feature
    morans_I_values = (N / W_sum.unsqueeze(-1)) * (numerator / denominator)  # [B, D]

    return morans_I_values
