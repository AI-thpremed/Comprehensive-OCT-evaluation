import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


    # The original netVLAD algorithm is developed for places recognition.
    # And it should be revised before using in video classification.
    # This revised netVLAD algorithm performs much better than original one
    # in video classification (more than 10 percent performance improvement),
    # And the revised netVLAD algorithm is provided by swliu, written by jeffhzhang.

class NetVLAD(nn.Module):
    """Creates a NetVLAD class.
    """

    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=False, add_batch_norm=True, is_training=True):
        print("cv fusion: netvlad...", flush=True)

        """Initialize a NetVLAD block.

        Args:
        feature_size: Dimensionality of the input features.
        max_samples: The maximum number of samples to pool.
        cluster_size: The number of clusters.
        output_dim: size of the output space after dimension reduction.
        add_batch_norm: (bool) if True, adds batch normalization.
        is_training: (bool) Whether or not the graph is training.
        """

        super(self.__class__, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

        self.gating_weights = torch.nn.Linear(
            in_features=output_dim, out_features=output_dim, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(num_features=output_dim)
        self.sigmoid1 = torch.nn.Sigmoid()

        self.cluster_weights = torch.nn.Linear(
            in_features=feature_size, out_features=cluster_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=cluster_size)
        self.softmax1 = torch.nn.Softmax(dim=1)

        self.cluster_weights2 = torch.nn.Parameter(
            torch.randn(1, feature_size, cluster_size))

        self.hidden1_weights = torch.nn.Linear(
            self.cluster_size*self.feature_size, self.output_dim)

    def forward(self, reshaped_input):
        """Forward pass of a NetVLAD block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' * 'max_samples' * 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' * 'feature_size'

        Returns:
        vlad: the pooled vector of size: 'batch_size' * 'output_dim'
        """
        reshaped_input = reshaped_input.view(-1, self.feature_size)

        activation = self.cluster_weights(reshaped_input)

        if self.add_batch_norm:
            activation = self.bn2(activation)

        activation = self.softmax1(
            activation).view(-1, self.max_samples, self.cluster_size)

        a_sum = torch.sum(activation, -2, keepdim=True)

        a = torch.mul(a_sum, self.cluster_weights2)

        activation = activation.permute(0, 2, 1)

        reshaped_input = reshaped_input.view(-1,
                                             self.max_samples, self.feature_size)

        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = vlad.contiguous().view(-1, self.cluster_size *
                                      self.feature_size)  # add contiguous
        vlad = F.normalize(vlad, p=2, dim=1)

        vlad = self.hidden1_weights(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

    def context_gating(self, input_layer):
        """Context Gating

        Args:
        input_layer: Input layer in the following shape:
        'batch_size' * 'feature_size'

        Returns:
        activation: gated layer in the following shape:
        'batch_size' * 'feature_size'
        """

        gates = self.gating_weights(input_layer)

        if self.add_batch_norm:
            gates = self.bn1(gates)

        gates = self.sigmoid1(gates)

        activation = torch.mul(input_layer, gates)

        return activation


class NetVLAD_v2(nn.Module):
    # based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=16, input_dim=2048, output_dim=2048,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            input_dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = input_dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(
            dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.fc = nn.Linear(num_clusters*input_dim, output_dim)

    def forward(self, x):
        """
            input  shape: [batch, max_samples, in_dim, 1] 
            output shape: [batch, out_dim] 
        """
        x = x.unsqueeze(-1)
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C],
                           dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[C:C+1, :].expand(
                    x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C+1, :].unsqueeze(2)
            vlad[:, C:C+1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        vlad = self.fc(vlad)

        return vlad



import torch
import torch.nn as nn
import torch.nn.functional as F

class NetVLAD_v3(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))  # 聚类中心，参见注释1
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )


    def forward(self, x):  # x: (N, C, H, W), H * W对应论文中的N表示局部特征的数目，C对应论文中的D表示特征维度
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim，使用L2归一化特征维度

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)  # (N, C, H, W)->(N, num_clusters, H, W)->(N, num_clusters, H * W)
        soft_assign = F.softmax(soft_assign, dim=1)  # (N, num_clusters, H * W)  # 参见注释3

        x_flatten = x.view(N, C, -1)  # (N, C, H, W) -> (N, C, H * W)
        
        # calculate residuals to each clusters
        # 减号前面前记为a，后面记为b, residual = a - b
        # a: (N, C, H * W) -> (num_clusters, N, C, H * W) -> (N, num_clusters, C, H * W)
        # b: (num_clusters, C) -> (H * W, num_clusters, C) -> (num_clusters, C, H * W)
        # residual: (N, num_clusters, C, H * W) 参见注释2
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

        # soft_assign: (N, num_clusters, H * W) -> (N, num_clusters, 1, H * W)
        # (N, num_clusters, C, H * W) * (N, num_clusters, 1, H * W)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)  # (N, num_clusters, C, H * W) -> (N, num_clusters, C)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten；vald: (N, num_clusters, C) -> (N, num_clusters * C)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad






class NetVLAD_v4(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad