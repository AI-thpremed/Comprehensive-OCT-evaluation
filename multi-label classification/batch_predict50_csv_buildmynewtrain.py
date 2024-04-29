import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import brier_score_loss,fbeta_score,recall_score
from sklearn.metrics import brier_score_loss,fbeta_score,recall_score,cohen_kappa_score
from sklearn.metrics import precision_score
import torch.nn as nn
from os import path
from PIL import Image
import numpy as np
import pandas as pd
# from resnet_withnon import resnet50
from resnet import resnet50
# from model import resnet50
# import cv2
import numpy as np
from torch.nn import DataParallel
import pandas as pd
from tqdm import tqdm
from dataloader.image_transforms import Image_Transforms
from fusion_model_buildmynewtrain import Build_MultiModel_ShareBackbone




data_res=[]


label_path='/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_octmulti/multi_label240126'

TEST_LISTS = ['train.csv']

IMAGE_PATH = '/vepfs/gaowh/oct/23octdata/'

this_path = "/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_octmulti/"



batch_size = 64


NON_IMAGE_IN_NAMES = []
# Must be same length as NON_IMAGE_IN_NAMES
# 1: category to number
# 0: do nothing
# (a, b): (entry - a)/b
NON_IMAGE_TRANSFORMS = []




path_join = path.join





def calculate_metrics_for_max_f2(key,data):
    # 提取患病概率和ground truth
    probabilities = data[key].astype(float)
    ground_truth = data['gt_'+key].astype(int)

    # 定义阈值范围和步长
    threshold_range = [i / 100 for i in range(101)]

    # 初始化最大 F2 指标、召回率、准确率和对应的阈值
    max_f2 = 0
    best_recall = 0
    best_accuracy = 0
    best_threshold = 0
    best_f1 = 0

    # 遍历阈值范围
    for threshold in threshold_range:
        # 将患病概率值转换为二进制预测结果
        predictions = probabilities.apply(lambda p: 1 if p >= threshold else 0)

        # 计算召回率、F2 指标和准确率
        recall = recall_score(ground_truth, predictions)
        f2 = fbeta_score(ground_truth, predictions, beta=2)
        accuracy = accuracy_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions)

        # 更新最大 F2 指标、召回率和对应的阈值
        if f2 > max_f2:
            max_f2 = f2
            best_recall = recall
            best_accuracy = accuracy
            best_threshold = threshold
            best_f1 = f1

    print("最佳阈值:", best_threshold)
    print("对应的最大召回率:", best_recall)
    print("最大 F2 指标:", max_f2)
    print("F1指标:", best_f1)

    print("对应的准确率:", best_accuracy)
    # 在最佳阈值下计算混淆矩阵
    predictions = probabilities.apply(lambda p: 1 if p >= best_threshold else 0)
    confusion_matrix = pd.crosstab(ground_truth, predictions, rownames=['Actual'], colnames=['Predicted'])
    print("混淆矩阵:")
    print(confusion_matrix)



def whatisa(key,data):
    # 提取患病概率和ground truth
    probabilities = data[key].astype(float)
    ground_truth = data['gt_'+key].astype(int)

    # 定义阈值范围和步长
    threshold_range = [i / 100 for i in range(101)]

    # 初始化最大F1指标和对应的阈值
    max_f1 = 0
    best_threshold = 0
    best_accuracy = 0
    best_recall = 0
    best_precision = 0

    # 遍历阈值范围
    for threshold in threshold_range:
        # 将患病概率值转换为二进制预测结果
        predictions = probabilities.apply(lambda p: 1 if p >= threshold else 0)

        # 计算F1指标
        f1 = f1_score(ground_truth, predictions)
        accuracy = accuracy_score(ground_truth, predictions)
        brier = brier_score_loss(ground_truth, probabilities)
        recall = recall_score(ground_truth, predictions)
        kappa = cohen_kappa_score(ground_truth, predictions)
        # precision = precision_score(ground_truth, predictions)

        # 更新最大F1指标和对应的阈值
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = threshold
            best_accuracy = accuracy
            best_brier_score = brier
            best_recall = recall
            best_kappa = kappa
            # best_precision = precision

    print("最佳阈值:", best_threshold)
    print("最大F1指标:", max_f1)
    print("准确率:", best_accuracy)
    print("Brier分数:", best_brier_score)
    print("对应的最大召回率:", best_recall)
    print("对应的kappa:", best_kappa)
    
    print("对应的精准度:", best_precision)

    # 在最佳阈值下计算混淆矩阵
    predictions = probabilities.apply(lambda p: 1 if p >= best_threshold else 0)
    confusion_matrix = pd.crosstab(ground_truth, predictions, rownames=['Actual'], colnames=['Predicted'])
    print("混淆矩阵:")
    print(confusion_matrix)







def write_csv(file_name: str, data: list,header:list) -> None:

    try:
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False, header=True)
        print(f"数据已成功写入 {file_name}。")
    except Exception as e:
        print(f"写入数据时发生错误: {e}")


def get_random_images(image_files, max_img, seed=123):
    total_frames = len(image_files)
    if total_frames <= max_img:
        if seed is not None:
            random.seed(seed)
        output = image_files.copy()
        if seed is not None:
            random.shuffle(output)
        while len(output) < max_img:
            output += image_files
        return sorted(output[:max_img])
    
    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(total_frames), max_img)
    indices.sort()
    image_list = [image_files[i] for i in indices]
    return image_list


class CustomDataset_Multi(torch.utils.data.Dataset):
    def __init__(self, im_dir, im_names, im_labels, im_id,im_transforms=None):
        self.im_dir = im_dir
        self.im_labels = im_labels
        self.im_names = im_names
        self.im_id=im_id
        if im_transforms:
            self.im_transforms = im_transforms
        else:
            self.im_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):

        return len(self.im_labels)

    def __getitem__(self, idx):

        image_list=get_random_images( self.im_names[idx].split(';'),10)

        modified_list = [os.path.join(self.im_dir, str(self.im_id[idx]), string) for string in image_list]
                

        images = []

        for image_path in modified_list:



            try:
                im = Image.open(image_path).convert('RGB')
                im = self.im_transforms(im)
                images.append(im)
            except:
                print('Error: Failed to open or verify image file {}'.format(image_path))


        return images, self.im_labels[idx],self.im_id[idx],image_list


def load_data_multi(label_path, train_lists, img_path,
              batchsize, classes,im_transforms,type):   


    train_sets = []
    train_loaders = []

    for train_list in train_lists:
        full_path_list = path_join(label_path,train_list)
        df = pd.read_csv(full_path_list)
        im_names = df['img_list'].to_numpy()
        im_labels = torch.tensor(df[classes].to_numpy(), dtype=torch.float)

        im_id=df['id'].to_numpy()


    
        train_sets.append(CustomDataset_Multi(img_path, im_names, im_labels , im_id,im_transforms))
        train_loaders.append(torch.utils.data.DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True,num_workers=8))
        print('Size for {0} = {1}'.format(train_list, len(im_names)))


    return train_loaders[0]



import random


def runpredict():

    modeltype="resnet152_avg"

    topk=3



    CLASSES = ['mh','mf','erm','cnv','me','rd']


    CLASSES = np.array(CLASSES)

    val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


    # Create training and test loaders
    validate_loader = load_data_multi(label_path, TEST_LISTS, IMAGE_PATH, batch_size,CLASSES, val_transforms,'test')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Build_MultiModel_ShareBackbone(CLASSES=CLASSES,backbone=modeltype)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    weights_path = '/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_octmulti/_resnet152_rd_withoutquality/resNet50_resnet152_rd_withoutquality_best.pth'

    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))


    data_res=[]


    results=[]


    df = pd.DataFrame(columns=['id', 'img_list','gt_mh', 'gt_mf', 'gt_erm', 'gt_cnv', 'gt_me','gt_rd', 'mh', 'mf', 'erm', 'cnv', 'me','rd'])
    # prediction
    model.eval()
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in validate_loader:

            val_images,val_labels,imgids,imgenames = val_data


            model = model.to('cuda')
            outputs = model(val_images)

            imgids = imgids.cpu().numpy()  
            val_labels = val_labels.cpu().numpy()  

            pure_list = [tensor.tolist() for tensor in outputs]


            # Append data to results
            for i in range(len(imgids)):
                    for j in range(10):
                        result = {
                            'id': imgids[i],
                            'img_list': imgenames[j][i],
                            'gt_mh': int(val_labels[i][0]),
                            'gt_mf': int(val_labels[i][1]),
                            'gt_erm': int(val_labels[i][2]),
                            'gt_cnv': int(val_labels[i][3]),
                            'gt_me': int(val_labels[i][4]),
                            'gt_rd': int(val_labels[i][5]),
                            'mh': round(pure_list[j][i][0],3),
                            'mf':  round(pure_list[j][i][1],3),
                            'erm':  round(pure_list[j][i][2],3),
                            'cnv': round(pure_list[j][i][3],3),
                            'me':round(pure_list[j][i][4],3),
                            'rd': round(pure_list[j][i][5],3)
                        }
                        results.append(result)


    df = pd.DataFrame(results)
    df.to_csv('train_data_stage1_results.csv', index=False)


def main():

 

    runpredict()




if __name__ == '__main__':
    main()
