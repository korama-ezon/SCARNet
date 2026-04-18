from functools import partial
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .gsa import GlobalSpatialAttention


class Gcn(nn.Module):
    def __init__(self, in_channels, out_channels, adj, keypoints, frames):
        super().__init__()
        self.adj = adj
        self.kernel_size = self.adj.size(0) 
        self.conv = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv, kvw->nctw', (x, self.adj))
        return x.contiguous()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        trunc_normal_(self.fc1.weight, std=.02)
        trunc_normal_(self.fc2.weight, std=.02)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_ln(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features)
        )
        self.act = act_layer()
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features)
        )
        self.drop = nn.Dropout(drop)
        trunc_normal_(self.fc1[0].weight, std=.02)
        trunc_normal_(self.fc2[0].weight, std=.02)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, length, frames, dim, tokens_dim, channels_dim, adj, keypoints, conv_large, gn_large, num_heads, sga_size, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group_layer=nn.GroupNorm):
        super().__init__()   
        self.norm1 = norm_layer(length)
        self.gcn_1 = Gcn(dim, dim, adj, keypoints, frames)

        self.dw_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=conv_large, padding=conv_large//2, groups=dim),
            group_layer(gn_large, dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1, padding=0, groups=1)
        )

        if frames == 1:
            self.mlp_1 = Mlp(in_features=length, hidden_features=tokens_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp_1 = Mlp_ln(in_features=length, hidden_features=tokens_dim, act_layer=act_layer, drop=drop)

        self.global_spatial_attn = GlobalSpatialAttention(
            channels=dim, num_joints=length, num_heads=num_heads, kernel_size=sga_size, frames=frames
        )
        
        self.gn_fuse = group_layer(gn_large, dim)
        self.act_fuse = act_layer()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.gcn_2 = Gcn(dim, dim, adj, keypoints, frames)
        
        self.mlp_2 = Mlp(in_features=dim, hidden_features=channels_dim, act_layer=act_layer, drop=drop)
        self.a_g = nn.Parameter(torch.ones(length)) 
        self.a_m = nn.Parameter(torch.ones(length)) 
        self.a_s = nn.Parameter(torch.ones(length)) 

        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

    def forward(self, x):
        res = x
        x = rearrange(x, 'b j c -> b c j')

        x = self.norm1(x)

        x_gcn_1 = rearrange(x, 'b c j -> b c 1 j')
        x_gcn_1 = self.gcn_1(x_gcn_1)
        x_gcn_1 = rearrange(x_gcn_1, 'b c 1 j -> b c j')

        x_mjem = self.dw_conv(x)
        x_mjem = self.mlp_1(x_mjem)

        x_sga = self.global_spatial_attn(x) 

        fused_spatial = self.gn_fuse(x_sga+ x_gcn_1 + x_mjem)

        x = self.drop_path(self.act_fuse(fused_spatial))

        x = rearrange(x, 'b c j -> b j c')
        res_channel = x

        x = self.norm2(x + res)
        mlp_output = self.mlp_2(x)
        
        x_gcn_2 = rearrange(mlp_output, 'b j c -> b c 1 j')
        x_gcn_2 = self.gcn_2(x_gcn_2)
        x_gcn_2 = rearrange(x_gcn_2, 'b c 1 j -> b j c')

        x = res_channel + self.drop_path(x_gcn_2)
        
        return x + res


class SCAF(nn.Module):
    def __init__(self, depth, embed_dim, channels_dim, tokens_dim, adj, keypoints, conv_large, gn_large, num_heads, sga_size, drop_rate=0.1, length=17, frames=1):
        super().__init__()
        drop_path_rate = 0.2 
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        group_layer = partial(nn.GroupNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(
                length=length, 
                frames=frames, 
                dim=embed_dim, 
                tokens_dim=tokens_dim, 
                channels_dim=channels_dim, 
                adj=adj, 
                drop=drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                keypoints=keypoints,
                conv_large=conv_large,
                gn_large=gn_large,
                num_heads=num_heads,
                sga_size=sga_size,
                group_layer=group_layer
            ) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
