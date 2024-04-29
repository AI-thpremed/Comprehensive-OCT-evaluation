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
class CVModel_Video_Dataset(Dataset):
    def __init__(self, label_list, transforms=None, train=False, val=False):
        self.train = train
        self.val = val
        self.transforms = transforms
        self.len_frames = 8

        videos = []
        for index, row in label_list.iterrows():
            videos.append((row["filename"], row["label"]))
        self.videos = videos

    def __getitem__(self, index):
        filename, label = self.videos[index]

        video = []
        zip_file = zipfile.ZipFile(filename)
        filelist = zip_file.namelist()
        filelist = [n for n in filelist if has_file_allowed_extension(n, '.jpg')]
        slicelist = framelist_tsn(filelist, self.len_frames, self.train)

        for imgfile in slicelist:
            img = Image.open(io.BytesIO(zip_file.read(imgfile))).convert('RGB')
            video.append(img)
        
        video = self.transforms(video)

        return video, label

    def __len__(self):
        return len(self.videos)


def collate_fn(batch):
    videos = []
    label = []
    for sample in batch:
        videos.append(sample[0])
        label.append(sample[1])

    return torch.stack(videos, 0), label


def get_files_fromtxt(root, mode):
    if mode == 'train' or mode == 'val':
        r_txt = open(root, 'r', encoding='utf-8')
        all_data_path, labels = [], []
        num_neg, num_pos = 0, 0
        for inf in r_txt:
            video_path = inf[:-1].split('\t')[0]
            label = inf[:-1].split('\t')[1]
            all_data_path.append(video_path)
            labels.append(int(label))
            if label == '0':
                num_neg += 1
            else:
                num_pos += 1

        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
     
        if mode == 'train':
            print('videos of train:{}'.format(len(labels)))
            print('videos of train_pos:{}'.format(num_pos))
            print('videos of train_neg:{}'.format(num_neg))
        elif mode == 'val':
            print('videos of valid:{}'.format(len(labels)))
            print('videos of valid_pos:{}'.format(num_pos))
            print('videos of valid_neg:{}'.format(num_neg))

        return all_files

    else:
        print('error')
