import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SEGating(nn.Module):

    def __init__(self, input_dim):

        super(SEGating, self).__init__()

        self.gating = nn.Sequential(
            nn.Linear(input_dim, input_dim//8),
            nn.BatchNorm1d(input_dim//8),
            nn.ReLU(),
            nn.Linear(input_dim//8, input_dim),
            nn.Sigmoid()
        )

    def forward(self, emb):
        mask = self.gating(emb)
        activation = emb * mask
        return activation

