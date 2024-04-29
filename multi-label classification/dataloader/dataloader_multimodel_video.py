import io
import zipfile
import random
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def sampling_tsn_v1(total, k, is_train):
    '''  if len(frames) <  K : random over sampling '''
    if is_train:
        if total >= k:
            frame_idx = np.array(random.sample(list(range(total)), k))
            frame_idx = np.sort(frame_idx)
        else:
            frame_idx = np.random.randint(low=0, high=total, size=(k - total,))
            frame_idx = np.sort(np.concatenate((np.arange(total), frame_idx)))
    else:
        if total >= k:
            frame_idx = int(total / k // 2)
            frame_idx += total / k * np.arange(k)
            frame_idx = np.array([int(t) for t in frame_idx])
        else:
            frame_idx = np.sort(np.concatenate((np.arange(total), np.floor(
                total/(k-total) * np.arange(k-total)).astype(np.int))))
    assert frame_idx.size == k
    return frame_idx

def framelist_tsn(c, k=3, is_train=True):
    total = len(c)
    frame_idx = sampling_tsn_v1(total, k, is_train)
    return [c[x] for x in frame_idx]


# 2.define dataset
class MultiModel_Video_Dataset(Dataset):
    def __init__(self, label_list, transforms=None, transforms_v=None, train=False, val=False):
        self.train = train
        self.val = val
        self.transforms = transforms
        self.transforms_v = transforms_v
        self.len_frames = 8

        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row["filename"], row["label"], row["filename_video"]))
        self.imgs = imgs

    def __getitem__(self, index):
        filename, label, filename_video = self.imgs[index]
        
        # image
        img = Image.open(filename).convert('RGB')
        img = self.transforms(img)
        
        # video
        video = []
        zip_file = zipfile.ZipFile(filename_video)
        filelist = zip_file.namelist()
        filelist = [n for n in filelist if has_file_allowed_extension(n, '.jpg')]
        slicelist = framelist_tsn(filelist, self.len_frames, self.train)
        for imgfile in slicelist:
            frame = Image.open(io.BytesIO(zip_file.read(imgfile))).convert('RGB')
            video.append(frame)
        video = self.transforms_v(video)

        return img, label, video

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    imgs = []
    label = []
    videos = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        videos.append(sample[2])

    return torch.stack(imgs, 0), label, torch.stack(videos, 0)


def get_files_fromtxt(root, mode):
    if mode == 'train' or mode =='val':
        r_txt = open(root, 'r', encoding='utf-8')
        all_data_path, labels, all_video_path = [], [], []
        num_neg, num_pos = 0, 0
        for inf in r_txt:
            img_path = inf[:-1].split('\t')[0]
            label = inf[:-1].split('\t')[1]
            video_path = inf[:-1].split('\t')[2]

            all_data_path.append(img_path)
            labels.append(int(label))
            all_video_path.append(video_path)

            if label == '0':
                num_neg += 1
            else:
                num_pos += 1

        all_files = pd.DataFrame({"filename": all_data_path, "label": labels, "filename_video": all_video_path})

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
