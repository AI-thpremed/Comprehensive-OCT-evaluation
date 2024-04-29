import os
import torch
import torch.optim as optim
import math
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
# import mmcv
import torch.nn.functional as F
from scipy.stats import pearsonr
import datetime
import time
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import defaultdict, deque
import torch.distributed as dist
from typing import List
# from torchvision import transforms
from PIL import Image
from torchvision import transforms as T
# from data.imageDataTransform import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize_and_RandomCrop, RandomCrop, ToTensor
from torchvision.utils import make_grid
from torch.autograd import Variable
from os.path import join
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms



class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        loss = -(1 - inputs) ** self.gamma * targets * torch.log(inputs) - inputs ** self.gamma * (1 - targets) * torch.log(1 - inputs)
        loss = loss.mean()
        return loss
