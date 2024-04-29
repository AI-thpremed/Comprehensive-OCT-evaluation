import os
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from resnet import resnet50_multi
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import DataParallel
import torch.nn.init as nn_init
from os import path
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import time
# from utils_loss import FocalLoss
from fusion_model import Build_MultiModel_ShareBackbone
import random
from dataloader.image_transforms import Image_Transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def compute_regression_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    corr = r2_score(targets, predictions)
    return mse, mae, rmse, corr

def writefile(name, list):
    f = open(name+'.txt', mode='w')
    for i in range(len(list)):
        s = str(list[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',') + '\n'
        f.write(s)
    f.close()



def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' success')
        return True
    else:
        return False


path_join = path.join




def get_random_images(image_files, max_img, seed=12):
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

        # names = self.im_names[idx].split(';')
        # middle_element = names[len(names) // 2]
        # image_list=[middle_element]

        modified_list = [os.path.join(self.im_dir, str(self.im_id[idx]), string) for string in image_list]
        # image_list_path=';'.join(modified_list)
        images = []
        for image_path in modified_list:
            try:
                im = Image.open(image_path).convert('RGB')
                im = self.im_transforms(im)
                images.append(im)
            except:
                print('Error: Failed to open or verify image file {}'.format(image_path))

        return images, self.im_labels[idx],self.im_id[idx]


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











def main():

    taskname='_resnet152_test2'
    modeltype="resnet152_avg"
    label_path='/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_octmulti/multi_label240126'
    path='/root/work2023/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet_octmulti/'+taskname

    IMAGE_PATH = '/vepfs/gaowh/oct/23octdata/'

    TRAIN_LISTS = ['train.csv']
    TEST_LISTS = ['test.csv']
    # #_resnet152_rd_withohegou
    CLASSES = ['mh','mf','erm','cnv','me','rd']
    CLASSES = np.array(CLASSES)
    mkdir(path)
    save_path_best = path+'/resNet152'+taskname+'_best.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    batch_size = 128
    # 4 load dataset
    train_transforms = Image_Transforms(mode='train', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
    val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()


    # Create training and test loaders
    validate_loader = load_data_multi(label_path, TEST_LISTS, IMAGE_PATH, batch_size,CLASSES, val_transforms,'test')

    train_loader = load_data_multi(label_path, TRAIN_LISTS, IMAGE_PATH, batch_size,CLASSES, train_transforms,'train')


    dfres = pd.read_csv(path_join(label_path, TEST_LISTS[0]))
    val_num=dfres.shape[0]

    dftrain = pd.read_csv(path_join(label_path, TRAIN_LISTS[0]))

    train_num=dftrain.shape[0]

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = Build_MultiModel_ShareBackbone(CLASSES=CLASSES,backbone=modeltype)

    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
    net.to(device)



    Loss_list = []
    Loss_list_val = []
    stat=[]

    epochs =30
    best_auroc =0.0
    train_steps = len(train_loader)




    val_steps=len(validate_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0

        val_loss=0.0
        train_bar = tqdm(train_loader)

        Ypred_train = []
        Ytruth_train = []
        for count, (data, target,_) in enumerate(train_bar):

            target = target.to(device)
            
            logits = net(data)

            loss = loss_function(logits, target.to(device))
 
            running_loss += loss.item()

            Ypred_train.append(logits.detach().cpu().numpy())
            Ytruth_train.append(target.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        Ypred1 = np.concatenate(Ypred_train, axis=0)
        Ytruth1 = np.concatenate(Ytruth_train, axis=0)
        try:
            auroc_ave = roc_auc_score(Ytruth1, Ypred1, average='weighted')
            auroc_classes = roc_auc_score(Ytruth1, Ypred1, average=None)
        except ValueError:
            print('WARNING: AUC undefined as only one sample is available for 1 or more of the classes')
            auroc_ave = 0.0
            auroc_classes = np.zeros(target.shape[1])
        print('Epoch {0}:  average auroc={1:g}'.format(epoch, auroc_ave))
        auroc_classes_dict = dict(zip(CLASSES, np.round(auroc_classes, 3)))
        print('Training Class auroc = ', auroc_classes_dict)


        # validate
        net.eval()
        imgid_list=[]
        Ypred = []
        Ytruth = []

        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images,val_labels,imgids = val_data
                outputs = net(val_images)

                Ypred.append(outputs.detach().cpu().numpy())
                Ytruth.append(val_labels.detach().cpu().numpy())

                imgid_list.append(list(imgids))


                loss_val = loss_function(outputs, val_labels.to(device))
                val_loss+=loss_val.item()

                val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                           epochs,loss_val)
        Ypred = np.concatenate(Ypred, axis=0)
        Ytruth = np.concatenate(Ytruth, axis=0)
        if epoch==epochs-1:
            writefile(path+"/Ypred",Ypred)
            writefile(path+"/Ytruth",Ytruth)
        try:
            val_auroc_ave = roc_auc_score(Ytruth, Ypred, average='weighted')
            val_auroc_classes = roc_auc_score(Ytruth, Ypred, average=None)
        except ValueError:
            print('WARNING: AUC undefined as only one sample is available for 1 or more of the classes')
            val_auroc_ave = 0.0
            val_auroc_classes = np.zeros(val_labels.shape[1])



        val_auroc_classes_dict = dict(zip(CLASSES, np.round(val_auroc_classes, 3)))
        print('Validation Class auroc = ', val_auroc_classes_dict)

        print("average:"+str(val_auroc_ave))

        stat.append(str(val_auroc_classes_dict)+"   average:"+str(val_auroc_ave))


        Loss_list.append(running_loss / train_steps)

        Loss_list_val.append(val_loss / val_steps)

        print('[epoch %d] train_loss: %.3f  val_loss: %.3f' %
              (epoch + 1, running_loss / train_steps, val_loss / val_steps))

        if val_auroc_ave > best_auroc:
            best_auroc = val_auroc_ave
            torch.save(net.state_dict(), save_path_best)

    print('Finished Training')
    stat.append(str(best_auroc))

    plt.subplot(1, 2, 2)
    x2 = range(0, epochs)
    y3 = Loss_list
    plt.plot(x2, y3, '-', label="Train Loss")
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(path+'/'+"loss"+taskname+".jpg")
    writefile(path+'/'+"Loss_list"+taskname, Loss_list)
    writefile(path+'/'+"Loss_list_val"+taskname, Loss_list_val)
    writefile(path+'/'+"stat_"+taskname, stat)


if __name__ == '__main__':
    main()