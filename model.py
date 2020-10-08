import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        #スッキプ結合
        self.residual=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        residual = self.residual(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv2(x)
        output = x+residual
        return output


class ResNet(nn.Module):
    def __init__(self, in_ch, f_out, n_ch):
        super(ResNet,self).__init__()
        self.layer1 = nn.Sequential(
            double_conv(in_ch, f_out, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            double_conv(f_out, f_out*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            double_conv(f_out*2, f_out*4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            double_conv(f_out*4, f_out*8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            double_conv(f_out*8, f_out*16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer6 = nn.Sequential(
            double_conv(f_out*16, f_out*32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer7 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.layer8 = nn.Sequential(
            nn.Linear(f_out*32, f_out*8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(f_out*8, n_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x=x.view(-1, 1024)
        x = self.layer8(x)

        return x




