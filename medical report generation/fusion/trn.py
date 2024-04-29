# the relation consensus module by Bolei
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from .average import SegmentConsensus


class RelationModuleMultiScale(torch.nn.Module):

    def __init__(self, img_feature_dim, num_frames=8, output_dim=512):
        super(RelationModuleMultiScale, self).__init__()
        print("cv fusion: multi-trn...", flush=True)

        assert num_frames in [8, 16]
        if num_frames == 8:
            self.scales = [3, 5]  # generate the multiple frame relations
        elif num_frames == 16:
            self.scales = [5, 8]  # generate the multiple frame relations
        self.img_feature_dim = img_feature_dim

        self.average = SegmentConsensus(img_feature_dim, output_dim)
        self.fc_fusion_scales = nn.ModuleList()  # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(nn.ReLU(), nn.Linear(
                scale * self.img_feature_dim, output_dim))
            self.fc_fusion_scales += [fc_fusion]

        self.relations_scales = []
        for scale in self.scales:
            relations_scale = self.get_index(num_frames, 2*scale)
            self.relations_scales.append(
                [relations_scale[0::2], relations_scale[1::2]])

        print('Multi-Scale Temporal Relation Network Module in use',
              ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):

        act_all = self.average(input)

        for scaleID in range(len(self.scales)):
            for idxs in self.relations_scales[scaleID]:
                act_relation = input[:, idxs, :]
                act_relation = act_relation.view(act_relation.size(
                    0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def get_index(self, total, k):
        # 均匀采样
        if total >= k:
            frame_idx = int(total / k // 2)
            frame_idx += total / k * np.arange(k)
            frame_idx = np.array([int(t) for t in frame_idx])
        else:
            frame_idx = np.sort(np.concatenate((np.arange(total), np.floor(
                total/(k-total) * np.arange(k-total)).astype(np.int))))
        return list(frame_idx)
