import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from einops import rearrange
from model.block.graph_frames import Graph
from model.block.scaf import SCAF
from model.block.AMRE import AMRE

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.graph = Graph('hm36_gt', 'spatial', pad=1)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False)
        
        self.embedding = nn.Linear(2*args.frames, args.channel)
        self.scaf = SCAF(args.layers, args.channel, args.d_hid, args.token_dim, self.A, args.keypoints, args.conv_large, args.gn_large, args.num_heads, args.sga_size,
                              length=args.n_joints, frames=args.frames)
        
        self.amre = AMRE(
            dim=args.channel
        )
        
        self.head = nn.Linear(args.channel, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> b j (c f)').contiguous()

        x = self.embedding(x)
        x = self.scaf(x)
        
        x = self.amre(x)
        
        x = self.head(x)

        x = rearrange(x, 'b j c -> b 1 j c').contiguous()

        return x
