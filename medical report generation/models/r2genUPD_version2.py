import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
# from modules.encoder_decoderUPD import EncoderDecoder
from modules.encoder_decoderUPDFusion import EncoderDecoder

import pdb

from fusion_model import Build_MultiModel_ShareBackbone

import torchvision.transforms as transforms
from resnet_multilabels import resnet50,resnet152,resnet101


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        CLASSES =  ['mh','mf','erm','cnv','me','rd']
        CLASSES = np.array(CLASSES)
        self.multilabel = resnet152(pretrained=True)
        self.multilabel.fc = nn.Sequential(nn.Linear(self.multilabel.fc.in_features, len(CLASSES)),
                                 nn.Sigmoid())
        model_weight_path = "/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_octmulti/_resnet152_org_top5/resNet50_resnet152_org_top5_best.pth"
        checkpoint = torch.load(model_weight_path)
        self.multilabel = nn.DataParallel(self.multilabel)
        self.multilabel.load_state_dict(checkpoint, strict=True)

        # model_weight_path = "/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_octmulti/_resnet152_org_top5/resNet50_resnet152_org_top5_best.pth"
        # self.multilabel.load_state_dict(torch.load(model_weight_path),strict=True)

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


    def forward_oct(self, images,backup, targets=None, mode='train'):

        att_feats_list = []
        fc_feats_list = []
        img_features=[]

        for i in range(images.size(1)):
            att_feats, fc_feats = self.visual_extractor(images[:, i])
            
            att_feats_list.append(att_feats)

            fc_feats_list.append(fc_feats)
            multlable = self.multilabel(images[:, i])

            img_features.append(multlable)

        avg_feats = torch.stack(img_features, dim=1)
        imglabels, _ = torch.max(avg_feats, dim=1)

        imglabels = imglabels.unsqueeze(2)
        threshold=0.9
        imglabels[imglabels >= threshold] = 1
        imglabels[imglabels < threshold] = 0

        # #method 1
        #
        imglabels = torch.nn.functional.pad(imglabels, (0, 0, 0, 49-6))  # 在第二维上填充个0，变成[batch_size, 49, 1]

        # #method 2
        # imglabels = imglabels.transpose(1, 2)  # 变成[batch_size,1, 9]
        # imglabels = imglabels.repeat(1, 49, 1)#新张量形状: torch.Size([bz, 49, 9])

        #method 3

        #method 4


        if mode == 'train':

            output = self.encoder_decoder(fc_feats_list, att_feats_list, imglabels,targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats_list, att_feats_list, imglabels,mode='sample')
        else:
            raise ValueError
        return output










