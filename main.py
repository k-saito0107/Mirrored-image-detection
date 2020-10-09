import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
import os.path as osp
from PIL import Image
from glob import glob

path = os.getcwd()
print(path)

#画像pathの取得
data_path = '/kw_resources/Mirrored-image-detection'
train_path = osp.abspath(data_path+'/normal_img_data/')
test_path = osp.abspath(data_path+'/abnormal_img_data/')

train_path_img = glob(osp.join(train_path,'*.png'))
test_path_img = glob(osp.join(test_path,'*.png'))

#データセットの作成

from make_dataset import Make_Dataset
from data_augumentation import Compose,  Scale, Resize


mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
scale = [0.7, 1.3]
width = 1920
height = 960
batch_size = 24

train_transform = transforms.Compose([
    Resize(width, height),
    Scale(scale),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


test_transform = transforms.Compose([
    Resize(width, height),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


train_dataset = Make_Dataset(img_path=train_path_img, img_transform=train_transform)
test_dataset = Make_Dataset(img_path=test_path_img, img_transform=test_transform)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=2)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True, num_workers=2)


from model import ResNet
in_ch = 3
f_out = 64
n_ch = 1

model = ResNet(in_ch, f_out, n_ch)



#モデルの学習
from train import train
num_epoch = 200

up_model = train(model, num_epoch, train_loader, test_loader)



print('finish')









