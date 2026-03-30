import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, 
                                   kernel_size=kernel_size, 
                                   groups=in_channels, 
                                   padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.relu(x)
        return x

class AMRE_Branch(nn.Module):
    def __init__(self, in_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.split_size = in_features // 4
        
        self.branch_3x3 = DepthwiseConv1d(self.split_size, kernel_size=3)
        self.branch_5x5 = DepthwiseConv1d(self.split_size, kernel_size=5)
        self.branch_7x7 = DepthwiseConv1d(self.split_size, kernel_size=7)
        self.branch_9x9 = DepthwiseConv1d(self.split_size, kernel_size=9)

        self.projector = nn.Conv1d(in_features, in_features, kernel_size=1)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        identity = x
        
        x_3x3 = x[:, :self.split_size, :]
        x_5x5 = x[:, self.split_size:2*self.split_size, :]
        x_7x7 = x[:, 2*self.split_size:3*self.split_size, :]
        x_9x9 = x[:, 3*self.split_size:, :]

        x_3x3 = self.branch_3x3(x_3x3) 
        x_5x5 = self.branch_5x5(x_5x5)
        x_7x7 = self.branch_7x7(x_7x7)
        x_9x9 = self.branch_9x9(x_9x9)

        x = torch.cat([x_3x3, x_5x5, x_7x7, x_9x9], dim=1) + identity
        
        identity = x
        x = self.projector(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return identity + x

class AMRE_Full(BaseModule):
    def __init__(self, dim, num_heads, factor=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.factor = factor

        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(dim))
        self.project1 = nn.Linear(dim, dim // factor)
        self.project2 = nn.Linear(dim // factor, dim)
        self.dropout = nn.Dropout(p=dropout)

        self.amre_branch = AMRE_Branch(
            in_features=dim // factor, 
            num_heads=num_heads
        )

    def forward(self, x):
        identity = x
        
        x = self.norm(x) * self.gamma + x * self.gammax
        x_amre = self.project1(x)

        x_amre = x_amre.permute(0, 2, 1)
        x_amre = self.amre_branch(x_amre)

        x_amre = x_amre.permute(0, 2, 1)
        x_amre = F.gelu(x_amre)
        x_amre = self.dropout(x_amre)
        x_amre = self.project2(x_amre)

        return identity + x_amre

class AMRE(nn.Module):
    def __init__(self, dim, num_heads=8, factor=8):
        super().__init__()
        self.amre_full = AMRE_Full(dim=dim, num_heads=num_heads, factor=factor)
    
    def forward(self, x):
        return self.amre_full(x)
