import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn as nn
from fusion.segating import SEGating
from fusion.average import project,SegmentConsensus
# from models.cv_models.swin_transformer import swin
# from models.cv_models.swin_transformer_v2 import swinv2
# # from models.cv_models.resnest import resnest50, resnest101
# from models.cv_models.convnext import convnext_tiny, convnext_small, convnext_base
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
        for i in range(len(x)):
            image = x[i]
            temp = self.model(image)
            encoder_output_list.append(temp)

        if self.backbone=="resnet152_avg":
            avg_feats = torch.stack(encoder_output_list, dim=1)
            summed_feats = torch.sum(avg_feats, dim=1)
            output = summed_feats / image.size(1)

        elif self.backbone=="resnet152_max":
            feats = torch.stack(encoder_output_list, dim=1)
            output, _ = torch.max(feats, dim=1)
        else:
            output = torch.cat(encoder_output_list, -1) # b, c1+c2
            if self.use_gate:
                output = self.gate(output)
            output = self.lg1(output)
            # output = self.cate2(output)
        output = self.lg2(output)
        return output

# from train_multi_avg import load_data_multi
# from dataloader.image_transforms import Image_Transforms
# from tqdm import tqdm
# if __name__ == '__main__':
#     model = Build_MultiModel_ShareBackbone()
#     label_path='/root/work2023/deep-learning-for-image-processing-master/data_set/TRSL_ALL'
#     IMAGE_PATH = '/vepfs/gaowh/tr_eyesl/'
#     TRAIN_LISTS = ['train.csv']
#     TEST_LISTS = ['test.csv']
#     val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
#     validate_loader = load_data_multi(label_path, TEST_LISTS, IMAGE_PATH, 16, val_transforms,'test')
#     val_bar = tqdm(validate_loader)
#     for val_data in val_bar:
#         val_images,val_labels,imgids = val_data
#         outputs = model(val_images)
#         print(outputs.shape)

