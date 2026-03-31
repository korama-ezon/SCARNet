from functools import partial
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .gsa import SpatialGlobalAttention


class Gcn(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = nn.Parameter(adj)
        self.kernel_size = self.adj.size(0)
        self.conv = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
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
    def __init__(self, length, frames, dim, tokens_dim, channels_dim, adj, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.frames = frames
        
        self.norm1 = norm_layer(length)
        self.gcn_1 = Gcn(dim, dim, adj)
        self.dw_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.GroupNorm(16, dim),
            nn.GELU()
        )
        self.mlp_1 = Mlp(in_features=length, hidden_features=tokens_dim, act_layer=act_layer, drop=drop)
        self.spatial_global_attn = SpatialGlobalAttention(
            channels=dim, num_joints=length, num_heads=6, kernel_size=9
        )
        self.gn_fuse = nn.GroupNorm(16, dim)
        self.act_fuse = act_layer()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.gcn_2 = Gcn(dim, dim, adj)
        self.mlp_2 = Mlp(in_features=dim, hidden_features=channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = rearrange(x, 'b j c -> b c j')
        res_spatial = x

        x = self.norm1(x)

        x_gcn_1 = rearrange(x, 'b c j -> b c 1 j')
        x_gcn_1 = self.gcn_1(x_gcn_1)
        x_gcn_1 = rearrange(x_gcn_1, 'b c 1 j -> b c j')

        #MJEM
        x_mjem = self.dw_conv(x)
        x_mjem = self.mlp_1(x_mjem)
        
        #GSA
        x_gsa = self.spatial_global_attn(x)

        fused_spatial = self.gn_fuse(x_gsa + x_gcn_1 + x_mjem)
        x = res_spatial + self.drop_path(self.act_fuse(fused_spatial))

        x = rearrange(x, 'b c j -> b j c')
        res_channel = x

        x = self.norm2(x)

        x_gcn_2 = rearrange(x, 'b j c -> b c 1 j')
        x_gcn_2 = self.gcn_2(x_gcn_2)
        x_gcn_2 = rearrange(x_gcn_2, 'b c 1 j -> b j c')

        mlp_output = self.mlp_2(x)

        mlp_output_4d = mlp_output.transpose(1, 2).unsqueeze(2)
        sca_output = self.sca_block(mlp_output_4d)
        sca_output = sca_output.squeeze(2).transpose(1, 2)

        x = res_channel + self.drop_path(sca_output + x_gcn_2)

        return x


class Scaf(nn.Module):
    def __init__(self, depth, embed_dim, channels_dim, tokens_dim, adj, drop_rate=0.1, length=17, frames=1):
        super().__init__()
        drop_path_rate = 0.2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

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
                norm_layer=norm_layer
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
