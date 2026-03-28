import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        
        assert kernel_size in (3, 5, 7, 9, 11), 
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = torch.cat([avg_out, max_out], dim=1)
        x_attn = self.conv1(x_attn)
        attention_weights = self.sigmoid(x_attn)
        
        weighted_features = attention_weights * x
        return weighted_features


class GlobalSpatialAttention(nn.Module):
    def __init__(self, channels, num_joints=17, num_heads=8, kernel_size=7, frames=1):
        super().__init__()
        self.channels = channels
        self.num_joints = num_joints
        self.frames = frames

        self.ln1 = nn.LayerNorm(channels)
        
        self.spatial_attn = SpatialAttention1D(kernel_size=kernel_size)

        self.dropout = nn.Dropout(p=0.5)
        if self.frames > 1:
            self.ln2 = nn.LayerNorm(num_joints)

        self.dw_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, padding=0, groups=1)
        )
        
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        
        self.proj_out = nn.Identity()
        

    def forward(self, x):
        B, C, J = x.shape
        assert J == self.num_joints, f"Joints mismatch: expected {self.num_joints}, got {J}"
        assert C == self.channels, f"Channels mismatch: expected {self.channels}, got {C}"
        
        residual = x

        x_kv = self.ln1(x.transpose(1, 2).contiguous())
        
        x_spatial = self.dropout(self.spatial_attn(x))

        if self.frames > 1:
            x_spatial = self.ln2(x_spatial)

        x_spatial = self.dw_conv(x_spatial) + residual
        
        x_seq = x_spatial.transpose(1, 2).contiguous()
        
        x_norm = self.ln1(x_seq)
        
        attn_output, _ = self.mha(query=x_norm, key=x_kv, value=x_kv, need_weights=False)
        
        x = attn_output + x_seq
        
        x = x.transpose(1, 2).contiguous()
        
        fo = self.proj_out(x)
        
        return fo