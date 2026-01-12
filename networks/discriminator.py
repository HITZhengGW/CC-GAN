import torch
import torch.nn as nn
from torch import cat
import torch.fft as fft


class FrequencyConv3d(nn.Module):
    # The core code will be released after the paper is officially accepted.


class Discriminator(torch.nn.Module):
    def __init__(self, device, channels=1, dim=128):
        super().__init__()
        self.pre_module = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.pool = nn.MaxPool3d(2, stride=2)

        self.frq_pre_module = nn.Sequential(
            FrequencyConv3d(in_channels=channels, out_channels=48, 
                            kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(48, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main_module = nn.Sequential(
            nn.Conv3d(in_channels=112, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),        

            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.output = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Tanh())


    def forward(self, x):
        xx = self.pre_module(x)
        xy = self.frq_pre_module(x)
        x = cat([xx, xy], dim=1)

        x = self.main_module(x)
        return self.output(x)
