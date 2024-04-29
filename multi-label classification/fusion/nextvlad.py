import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class NextVLAD(nn.Module):
    """NextVLAD layer implementation"""

    def __init__(self, feature_size, max_frames, cluster_size, output_dim,
                 expansion=2, groups=8, gating=False):
        super(NextVLAD, self).__init__()
        print("cv fusion: nextvlad...", flush=True)

        self.feature_size = feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups
        self.new_feature_size = expansion * feature_size // groups

        # for dim expansion
        self.expand_weights = nn.Parameter(
            torch.Tensor(feature_size, expansion * feature_size))
        self.expand_biases = nn.Parameter(
            torch.Tensor(expansion * feature_size))
        # for group attention
        self.group_att_weights = nn.Parameter(
            torch.Tensor(expansion * feature_size, groups))
        self.group_att_biases = nn.Parameter(torch.Tensor(groups))
        # for cluster weights
        self.cluster_weights = nn.Parameter(torch.Tensor(
            expansion * feature_size, groups * cluster_size))
        self.cluster_weights2 = nn.Parameter(
            torch.Tensor(1, self.new_feature_size, cluster_size))

        output_feature_size = self.new_feature_size * cluster_size
        self.batchnorm = nn.BatchNorm1d(num_features=output_feature_size)
        self._init_params()
        self.output_dim = output_dim
        self.hidden1_weights = nn.Linear(
            self.cluster_size * self.new_feature_size, self.output_dim)

        self.gating = gating
        self.gating_weights = nn.Linear(
            in_features=output_dim, out_features=output_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=output_dim)
        self.sigmoid1 = nn.Sigmoid()

    def _init_params(self):
        stdv = 1. / math.sqrt(self.expand_weights.size(1))
        self.expand_weights.data.uniform_(-stdv, stdv)
        self.expand_biases.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.group_att_weights.size(1))
        self.group_att_weights.data.uniform_(-stdv, stdv)
        self.group_att_biases.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.cluster_size)
        self.cluster_weights.data.uniform_(-stdv, stdv)
        self.cluster_weights2.data.uniform_(-stdv, stdv)

    def context_gating(self, input_layer):
        gates = self.gating_weights(input_layer)
        gates = self.bn1(gates)
        gates = self.sigmoid1(gates)
        activation = torch.mul(input_layer, gates)
        return activation

    def forward(self, ftr_input):
        """
        ftr_input shape: batch_size * max_frame * feature_size
        """
        assert ftr_input.size(2) == self.feature_size
        assert ftr_input.size(1) == self.max_frames

        # dim expansion
        # shape: batch_size * max_frame * (feature_size*expansion)
        ftr_input = torch.matmul(
            ftr_input, self.expand_weights) + self.expand_biases
        # attention   # shape: batch_size * max_frame * groups
        attention = torch.matmul(
            ftr_input, self.group_att_weights) + self.group_att_biases
        attention = torch.sigmoid(attention)
        attention = attention.view(-1, self.max_frames * self.groups, 1)

        # shape: (batch_size*max_frame) * (feature_size*expansion)
        reshaped_input = ftr_input.view(-1, self.expansion * self.feature_size)
        # shape: (batch_size*max_frame) * (groups*cluster_size)
        activation = torch.matmul(reshaped_input, self.cluster_weights)
        #activation = self.batchnorm1(activation)
        activation = activation.view(-1, self.max_frames *
                                     self.groups, self.cluster_size)
        # batch_size* (max_frame * groups)*cluster_size
        activation = torch.softmax(activation, dim=-1) * attention
        # shape: batch_size * 1 * cluster_size
        a_sum = torch.sum(activation, dim=-2, keepdim=True)
        # shape: batch_size * new_feature_size * cluster_size
        a = a_sum * self.cluster_weights2

        # shape: batch_size * cluster_size * (max_frame*groups)
        activation = activation.permute(0, 2, 1)

        reshaped_input = ftr_input.view(-1, self.max_frames *
                                        self.groups, self.new_feature_size)
        # shape: batch_size * new_feature_size * cluster_size
        vlad = torch.matmul(activation, reshaped_input).permute(0, 2, 1) - a

        #vlad = F.normalize(vlad, p=2, dim=1).view(-1, self.cluster_size * self.new_feature_size)
        vlad = F.normalize(vlad, p=2, dim=1).contiguous(
        ).view(-1, self.cluster_size * self.new_feature_size)
        vlad = self.batchnorm(vlad)

        # new added in 2020/01/06
        vlad = self.hidden1_weights(vlad)
        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad
