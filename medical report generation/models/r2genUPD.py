import torch
import torch.nn as nn
import numpy as np
from modules.visual_extractor import VisualExtractor
# from modules.encoder_decoderUPD import EncoderDecoder
from modules.encoder_decoderUPDFusion import EncoderDecoder
import pdb
from fusion_model import Build_MultiModel_ShareBackbone
import torchvision.transforms as transforms


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)


        
        CLASSES = ['quality','mh','mf','erm','vma','ao','rd','cnv','me']
        CLASSES = np.array(CLASSES)

        self.multilabel = Build_MultiModel_ShareBackbone(CLASSES=CLASSES,backbone='resnet152_avg')



        model_weight_path = "/data/gaowh/work/24process/use_image_captioning-main/oct/multi_label/resNet50_resnet152_multi2test_best.pth"
        self.multilabel.load_state_dict(torch.load(model_weight_path),strict=False)




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

        # 循环处理每个图像
        for i in range(images.size(1)):
            # 提取特征
            att_feats, fc_feats = self.visual_extractor(images[:, i])
            
            att_feats_list.append(att_feats)

            fc_feats_list.append(fc_feats)



        backupimg=backup

        backupimg = torch.unbind(backupimg, 1)



        imglabels=self.multilabel(backupimg)

        imglabels = imglabels.unsqueeze(2)  # 变成[batch_size, 9, 1]
        threshold=0.9
        imglabels[imglabels >= threshold] = 1
        imglabels[imglabels < threshold] = 0


        # #method 1
        #
        imglabels = torch.nn.functional.pad(imglabels, (0, 0, 0, 49-9))  # 在第二维上填充个0，变成[batch_size, 49, 1]


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










