from torchvision import transforms as T
from dataloader.image_transforms_helper import MultiScaleCrop


class Image_Transforms(object):
    def __init__(self, mode='train', dataloader_id=1, square_size=256, crop_size=224):
        self.mode = mode
        self.dataloader_id = dataloader_id
        self.square_size = square_size
        self.crop_size = crop_size
        
    def build_train_transforms(self):
        train_transforms_1 = T.Compose([
            T.Resize((self.square_size, self.square_size)),
            T.RandomCrop((self.crop_size, self.crop_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        train_transforms_2 = T.Compose([
            T.Resize(self.square_size),
            T.RandomCrop((self.crop_size, self.crop_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        train_transforms_3 = T.Compose([
            MultiScaleCrop(self.crop_size, [1, .875, .75, .66]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        train_transforms_4 = T.Compose([
            T.Resize([self.crop_size, self.crop_size]),
            T.RandomResizedCrop(self.crop_size, scale=(0.6, 1.0), ratio=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        train_transforms_5 = T.Compose([
            T.Resize([self.crop_size, self.crop_size]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        train_transforms_all = [train_transforms_1,train_transforms_2,train_transforms_3,train_transforms_4,
                                train_transforms_5]
        return train_transforms_all[self.dataloader_id-1]

    def build_val_transforms(self):
        val_transforms_1 = T.Compose([
            T.Resize((self.square_size, self.square_size)),
            T.CenterCrop((self.crop_size, self.crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        val_transforms_2 = T.Compose([
            T.Resize(self.square_size),
            T.CenterCrop((self.crop_size, self.crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        val_transforms_3 = T.Compose([
            T.Resize(self.square_size),
            T.CenterCrop((self.crop_size, self.crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        val_transforms_4 = T.Compose([
            T.Resize([self.crop_size, self.crop_size]),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        val_transforms_5 = T.Compose([
            T.Resize([self.crop_size, self.crop_size]),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        val_transforms_all = [val_transforms_1, val_transforms_2, val_transforms_3, val_transforms_4,
                              val_transforms_5]
        return val_transforms_all[self.dataloader_id-1]


    def get_transforms(self):
        print('bulid dataloader, mode:', self.mode,', dataloader_id:',self.dataloader_id)
        if self.mode == 'train':
            return self.build_train_transforms()
        else:
            return self.build_val_transforms()







# class Image_Transforms(object):
#     def __init__(self, mode='train', dataloader_id=1, square_size=256, crop_size=224):
#         self.mode = mode
#         self.dataloader_id = dataloader_id
#         self.square_size = square_size
#         self.crop_size = crop_size
        
#     def build_train_transforms(self):
#         train_transforms_1 = T.Compose([
#             T.Resize((self.square_size, self.square_size)),
#             T.CenterCrop((self.crop_size, self.crop_size)),
#             T.HorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         train_transforms_2 = T.Compose([
#             T.Resize(self.square_size),
#             T.CenterCrop((self.crop_size, self.crop_size)),
#             T.HorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         train_transforms_3 = T.Compose([
#             MultiScaleCrop(self.crop_size, [1, .875, .75, .66]),
#             T.HorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         train_transforms_4 = T.Compose([
#             T.Resize([self.crop_size, self.crop_size]),
#             T.RandomResizedCrop(self.crop_size, scale=(0.6, 1.0), ratio=(0.8, 1.0)),
#             T.HorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         train_transforms_5 = T.Compose([
#             T.Resize([self.crop_size, self.crop_size]),
#             T.HorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         train_transforms_all = [train_transforms_1,train_transforms_2,train_transforms_3,train_transforms_4,
#                                 train_transforms_5]
#         return train_transforms_all[self.dataloader_id-1]

#     def build_val_transforms(self):
#         val_transforms_1 = T.Compose([
#             T.Resize((self.square_size, self.square_size)),
#             T.CenterCrop((self.crop_size, self.crop_size)),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         val_transforms_2 = T.Compose([
#             T.Resize(self.square_size),
#             T.CenterCrop((self.crop_size, self.crop_size)),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         val_transforms_3 = T.Compose([
#             T.Resize(self.square_size),
#             T.CenterCrop((self.crop_size, self.crop_size)),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         val_transforms_4 = T.Compose([
#             T.Resize([self.crop_size, self.crop_size]),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         val_transforms_5 = T.Compose([
#             T.Resize([self.crop_size, self.crop_size]),
#             T.ToTensor(),
#             T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
#         val_transforms_all = [val_transforms_1, val_transforms_2, val_transforms_3, val_transforms_4,
#                               val_transforms_5]
#         return val_transforms_all[self.dataloader_id-1]


#     def get_transforms(self):
#         print('bulid dataloader, mode:', self.mode,', dataloader_id:',self.dataloader_id)
#         if self.mode == 'train':
#             return self.build_train_transforms()
#         else:
#             return self.build_val_transforms()













if __name__ == '__main__':
    from PIL import Image
    img = Image.open('E:/data/glaucoma_datasets/glaucoma_cls/ORIGA/JPEGImages/001.jpg')
    train_transforms = Image_Transforms(mode='train', dataloader_id=1).get_transforms()
    val_transforms = Image_Transforms(mode='val', dataloader_id=1).get_transforms()

    img_t = train_transforms(img)
    img_t.show()
