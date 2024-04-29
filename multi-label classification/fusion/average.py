import os
import sys
import torch
import torch.nn as nn

class SegmentConsensus(nn.Module):
    def __init__(self, in_features=2048, out_features=256):
        print("cv fusion: average...", flush=True)
        super(SegmentConsensus, self).__init__()
        self.linear_logits = torch.nn.Linear(
            in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.linear_logits(x)
        return x

# class SegmentConsensus(nn.Module):
#     def __init__(self, in_features=40960, out_features=256):
#         print("cv fusion: average...", flush=True)
#         super(SegmentConsensus, self).__init__()
#         self.linear_logits = torch.nn.Linear(
#             in_features=in_features, out_features=out_features)

#     def forward(self, x):
#         x = self.linear_logits(x)
#         out = x.mean(dim=1)
#         return out



class project(nn.Module):
    def __init__(self, in_features=2048, out_features=256):
        print("cv fusion: project...", flush=True)
        super(project, self).__init__()
        self.inter = nn.Linear(in_features, in_features*4)
        self.act = nn.GELU()
        self.output = nn.Linear(in_features*4, out_features)

    def forward(self, x):
        x = self.inter(x)
        x = self.act(x)
        x = self.output(x)
        return x
