import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 2.define dataset
class CVModel_Dataset(Dataset):
    def __init__(self, label_list, transforms=None, train=False, val=False):
        self.train = train
        self.val = val
        self.transforms = transforms

        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row["filename"], row["label"]))
        self.imgs = imgs

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = Image.open(filename).convert('RGB')
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), label


def get_files_fromtxt(root, mode):
    if mode == 'train' or mode == 'val':
        r_txt = open(root, 'r', encoding='utf-8')
        all_data_path, labels = [], []
        num_neg, num_pos = 0, 0
        for inf in r_txt:
            img_path = inf[:-1].split('\t')[0]
            label = inf[:-1].split('\t')[1]
            all_data_path.append(img_path)
            labels.append(int(label))
            if label == '0':
                num_neg += 1
            else:
                num_pos += 1

        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
     
        if mode == 'train':
            print('imgs of train:{}'.format(len(labels)))
            print('imgs of train_pos:{}'.format(num_pos))
            print('imgs of train_neg:{}'.format(num_neg))
        elif mode == 'val':
            print('imgs of valid:{}'.format(len(labels)))
            print('imgs of valid_pos:{}'.format(num_pos))
            print('imgs of valid_neg:{}'.format(num_neg))

        return all_files

    else:
        print('error')
