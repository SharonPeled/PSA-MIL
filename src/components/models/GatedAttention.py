import torch.nn as nn
from torch.nn.functional import softmax


# copy paste from https://github.com/hrzhang1123/DTFD-MIL
class Attention_Gated(nn.Module):
    def __init__(self, L, D=128, K=1, softmax_dim=0):
        super(Attention_Gated, self).__init__()
        self.softmax_dim = softmax_dim

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = softmax(A, dim=self.softmax_dim)  # softmax over N
        return A


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class ResidualFullyConnected(nn.Module):
    def __init__(self, n_channels, m_dim, numLayer_Res):
        super(ResidualFullyConnected, self).__init__()
        self.num_layers = numLayer_Res
        if self.num_layers != 0:
            self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
            self.relu1 = nn.ReLU(inplace=True)
            numLayer_Res -= 1
            self.numRes = numLayer_Res
            if self.numRes > 0:
                self.resBlocks = []
                for ii in range(numLayer_Res):
                    self.resBlocks.append(residual_block(m_dim))
                self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        if self.num_layers == 0:
            return x
        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x