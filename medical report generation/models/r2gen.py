import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
import pdb


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args.dataset_name=='oct':
            self.forward = self.forward_oct

        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output



    def forward_oct(self, images, targets=None, mode='train'):

    
        att_feats_list = []
        fc_feats_list = []

        # 循环处理每个图像
        for i in range(images.size(1)):
            # 提取特征
            att_feats, fc_feats = self.visual_extractor(images[:, i])
            
            att_feats_list.append(att_feats)

            fc_feats_list.append(fc_feats)
        # pdb.set_trace()
        avg_feats = torch.stack(att_feats_list, dim=1)
        summed_feats = torch.sum(avg_feats, dim=1)  # 逐个特征相加
        avg_feats_mean = summed_feats / images.size(1)  # 计算平均特征


        avg_fc_feats = torch.stack(fc_feats_list, dim=1)
        summed_fc_feats = torch.sum(avg_fc_feats, dim=1)  # 逐个特征相加
        avg_fc_feats_mean = summed_fc_feats / images.size(1)  # 计算平均特征



        if mode == 'train':
            output = self.encoder_decoder(avg_fc_feats_mean, avg_feats_mean, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_fc_feats_mean, avg_feats_mean, mode='sample')
        else:
            raise ValueError
        return output










