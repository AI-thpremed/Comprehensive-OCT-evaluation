import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn as nn
from fusion.segating import SEGating
from fusion.average import project,SegmentConsensus

from resnet import resnet50,resnet50_single_cbma,resnet152


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
        
# img+img share backbone
class Build_MultiModel_ShareBackbone(nn.Module):
    def __init__(self, CLASSES,backbone='resnet152', input_dim=2048, num_classes=1, use_gate=True, pretrained_modelpath='None'):
        super().__init__()
        self.input_dim = input_dim*10
        # self.num_classes = num_classes
        self.use_gate = use_gate
        self.gate = SegmentConsensus(self.input_dim,2048*5)
        self.lg1 = torch.nn.Linear(in_features=2048*5, out_features=2048)
        self.lg2 = nn.Sequential(nn.Linear(self.lg1.out_features, len(CLASSES)),nn.Sigmoid())
        
        self.backbone=backbone

        self.model = resnet152()
        self.model.fc = Identity()

        # if backbone=="resnet50_single_cbma":
        #     self.model = resnet50_single_cbma()
        # elif backbone=="resnet152":
        #     self.model = resnet152()

        #     self.model.fc = Identity()



        print('init model:', backbone)

        if self.use_gate:
            print('use gate')
        else:
            print('no use gate')


    def forward(self, x):
        encoder_output_list = []
        temp_output_list=[]
        # 循环处理每个图像
        for i in range(len(x)):
            image = x[i]
            temp = self.model(image)
            encoder_output_list.append(temp)
            temp_output = self.lg2(temp)
            temp_output_list.append(temp_output)
        if self.backbone=="resnet152_avg":
            avg_feats = torch.stack(encoder_output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
            output = summed_feats / image.size(1)  # 计算平均特征
        elif self.backbone=="resnet152_max":
            feats = torch.stack(encoder_output_list, dim=1)
            output, _ = torch.max(feats, dim=1)
        else:
            output = torch.cat(encoder_output_list, -1) # b, c1+c2
            if self.use_gate:
                output = self.gate(output)
            output = self.lg1(output)
        return temp_output_list
