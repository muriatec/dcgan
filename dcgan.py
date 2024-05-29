import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super.__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4), # input z: 100 x 1
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True), # output size: 1024 x 4 x 4
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True), # output size: 512 x 8 x 8,
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True), # output size: 256 x 16 x 16,
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True), # output size: 128 x 32 x 32,
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        

    def forward(self, x):
        return self.main(x)