import os
import sys
import torch
import torch.nn as nn


#这里是两个方案，一个是高维转低维。还有一个是project就是翻倍映射再降维。
#下面是我的做法，如果不这么做。系统会学习率爆炸。然后所有都是Nan了。必须缓慢一点降维
# self.gate = SegmentConsensus(self.input_dim,2048*5)
# self.lg1 = torch.nn.Linear(in_features=2048*5, out_features=2048)
# self.lg2 = torch.nn.Linear(in_features=2048, out_features=self.num_classes)


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
