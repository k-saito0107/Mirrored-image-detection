import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
from PIL import Image

path = os.getcwd()
print(path)
#データセットの作成

from torchvision.datasets import ImageFolder
from data_augumentation import Compose,  Scale, Resize


mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
scale = [0.7, 1.3]
width = 960
height = 512
batch_size = 24

train_transform = transforms.Compose([
    Resize(width, height),
    Scale(scale),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    Resize(width, height),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


train_images=ImageFolder(
    '/kw_resources/Mirrored-image-detection/data/train/',
    transform=train_transform
)

test_images=ImageFolder(
    '/kw_resources/Mirrored-image-detection/data/test/',
    transform=val_transform
)

train_loader=torch.utils.data.DataLoader(train_images,batch_size=batch_size,shuffle=True, num_workers=2)
test_loader=torch.utils.data.DataLoader(test_images,batch_size=batch_size,shuffle=True, num_workers=2)


from model import ResNet
in_ch = 3
f_out = 32
n_ch = 2

model = ResNet(in_ch, f_out, n_ch)



#モデルの学習
from train import train
num_epoch = 200

up_model = train(model, num_epoch, train_loader, test_loader)



print('finish')









