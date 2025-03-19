import torch
import torch.nn as nn


class DistanceScaler(nn.Module):
    def __init__(self, scaler_type, num_heads):
        super(DistanceScaler, self).__init__()
        self.scaler_type = scaler_type
        self.num_heads = num_heads
        self.param = nn.Parameter(torch.ones(num_heads))
        self.bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, distance):
        if self.scaler_type is None:
            return distance
        distance = torch.stack([distance for _ in range(self.num_heads)], dim=1)
        param = torch.clamp(self.param, min=1e-6).view(1, self.num_heads, 1, 1)  # [1, num_heads, 1, 1]
        bias = torch.clamp(self.bias, min=1e-6).view(1, self.num_heads, 1, 1)
        if self.scaler_type == 'Linear':
            return param * distance + bias
        if self.scaler_type == 'Log':
            return (param * (distance+1)).log() + bias
        if self.scaler_type == 'Log_NON_LEARNABLE':
            return (distance+1).log()
        if self.scaler_type == 'Poly':
            return distance**param + bias
        raise NotImplementedError


