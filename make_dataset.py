import torch
import torch.utils.data as data
from PIL import Image
import torchvision
import torchvision.transforms as transforms


class Make_Dataset(data.Dataset):
    def __init__(self, img_path, img_transform):
        self.img_path = img_path
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):
        img_file_path = self.img_path[index]
        img = Image.open(img_file_path)
        img = self.img_transform(img)

        return img