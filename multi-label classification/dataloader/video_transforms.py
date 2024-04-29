from torchvision import transforms as T
from dataloader.video_transforms_helper import GroupMultiScaleCrop, GroupRandomHorizontalFlip, GroupCenterCrop
from dataloader.video_transforms_helper import GroupSquareResize, GroupRandomCrop, GroupRandomSizedCrop
from dataloader.video_transforms_helper import Stack, ToTorchFormatTensor, GroupNormalize, GroupScale


class Video_Transforms(object):
    def __init__(self, mode='train', dataloader_id=1, square_size=256, crop_size=224):
        self.mode = mode
        self.dataloader_id = dataloader_id
        self.square_size = square_size
        self.crop_size = crop_size
        
    def build_train_transforms(self):
        train_transforms_1 = T.Compose([
                    GroupSquareResize((self.square_size, self.square_size)),
                    GroupRandomCrop((self.crop_size, self.crop_size)),
                    GroupRandomHorizontalFlip(is_flow=False),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        train_transforms_2 = T.Compose([
                    GroupScale(self.square_size),
                    GroupRandomCrop((self.crop_size, self.crop_size)),
                    GroupRandomHorizontalFlip(is_flow=False),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        train_transforms_3 = T.Compose([
                    GroupMultiScaleCrop(self.crop_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        train_transforms_4 = T.Compose([
                    GroupSquareResize((self.crop_size, self.crop_size)),
                    GroupRandomSizedCrop(self.crop_size),
                    GroupRandomHorizontalFlip(is_flow=False),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        train_transforms_5 = T.Compose([
                    GroupSquareResize((self.crop_size, self.crop_size)),
                    GroupRandomHorizontalFlip(is_flow=False),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        train_transforms_all = [train_transforms_1,train_transforms_2,train_transforms_3,train_transforms_4,
                                train_transforms_5]
        return train_transforms_all[self.dataloader_id-1]

    def build_val_transforms(self):
        val_transforms_1 = T.Compose([
                    GroupSquareResize((self.square_size, self.square_size)),
                    GroupCenterCrop((self.crop_size, self.crop_size)),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        val_transforms_2 = T.Compose([
                    GroupScale(self.square_size),
                    GroupCenterCrop((self.crop_size, self.crop_size)),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        val_transforms_3 = T.Compose([
                    GroupScale(self.square_size),
                    GroupCenterCrop((self.crop_size, self.crop_size)),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        val_transforms_4 = T.Compose([
                    GroupSquareResize((self.crop_size, self.crop_size)),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        val_transforms_5 = T.Compose([
                    GroupSquareResize((self.crop_size, self.crop_size)),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        val_transforms_all = [val_transforms_1, val_transforms_2, val_transforms_3, val_transforms_4,
                              val_transforms_5]
        return val_transforms_all[self.dataloader_id-1]


    def get_transforms(self):
        print('bulid dataloader, mode:', self.mode,', dataloader_id:',self.dataloader_id)
        if self.mode == 'train':
            return self.build_train_transforms()
        else:
            return self.build_val_transforms()
