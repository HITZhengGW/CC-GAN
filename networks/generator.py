from torch import nn
from torch import cat
from mamba_ssm import Mamba
import torch

class O2Mamba(nn.Module):
    # The core code will be released after the paper is officially accepted.
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(ConvBlock, self).__init__()

        insert_channels = out_channels #if in_channels > out_channels else out_channels // 2
        layers = [
            nn.Conv3d(in_channels, insert_channels, 3, stride=1, padding='same'),
            nn.ReLU(True),
            nn.Conv3d(insert_channels, out_channels, 3, stride=1, padding='same'),
            nn.ReLU(True)
        ]
        if batch_norm:
            layers.insert(1, nn.InstanceNorm3d(insert_channels, affine=True))
            layers.insert(len(layers)-1, nn.InstanceNorm3d(out_channels, affine=True)) 
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(Down, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)  # conv
        out = self.pool(x)  # down
        return x, out

class MSAF(nn.Module):
    # The core code will be released after the paper is officially accepted.


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=False):
        super(Up, self).__init__()
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest') 
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

        self.conv_block = ConvBlock(in_channels + in_channels//2, out_channels, batch_norm)
        self.skip = MSAF(in_channels, in_channels//2, in_channels + in_channels//2)

    def forward(self, x, conv):
        x = self.sample(x)  # up
        x = self.skip(x, conv)
        x = self.conv_block(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=1, num_filters=48, class_num=1, batch_norm=True, sample=False):
        super(Generator, self).__init__()

        self.down1 = Down(in_channels, num_filters, batch_norm)
        self.down2 = Down(num_filters, num_filters * 2, batch_norm)
        self.down3 = Down(num_filters * 2, num_filters * 4, batch_norm)
        self.m1 = O2Mamba(num_filters * 4)
        self.down4 = Down(num_filters * 4, num_filters * 8, batch_norm)
        self.m2 = O2Mamba(num_filters * 8)

        self.neck = ConvBlock(num_filters * 8, num_filters * 16, batch_norm)

        self.up1 = Up(num_filters * 16, num_filters * 8, batch_norm, sample)
        self.m3 = O2Mamba(num_filters * 8)
        self.up2 = Up(num_filters * 8, num_filters * 4, batch_norm, sample)
        self.m4 = O2Mamba(num_filters * 4)
        self.up3 = Up(num_filters * 4, num_filters * 2, batch_norm, sample)
        self.up4 = Up(num_filters * 2, num_filters * 1, batch_norm, sample)

        self.conv_class = nn.Conv3d(num_filters * 1, class_num, 1, stride=1, padding='same')

    def forward(self, x):
        conv1, x = self.down1(x)
        conv2, x = self.down2(x)
        conv3, x = self.down3(x)
        x = self.m1(x)
        conv4, x = self.down4(x)
        x = self.m2(x)

        x = self.neck(x)

        x = self.up1(x, conv4)
        x = self.m3(x)
        x = self.up2(x, conv3)
        x = self.m4(x)
        x = self.up3(x, conv2)
        x = self.up4(x, conv1)

        out = self.conv_class(x)

        return out