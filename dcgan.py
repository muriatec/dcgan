import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=128, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8, kernel_size=6), # input z: 100 x 1
            nn.BatchNorm2d(num_features=ngf*8),
            nn.ReLU(inplace=True), # output size: 1024 x 6 x 6
            nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=ngf*4),
            nn.ReLU(inplace=True), # output size: 512 x 12 x 12
            nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=ngf*2),
            nn.ReLU(inplace=True), # output size: 256 x 24 x 24
            nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True), # output size: 128 x 48 x 48
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1),
            # not applying batchnorm to the generator output layer
            nn.Tanh() # output size: 3 x 96 x 96
        )
        
    def forward(self, x):
        return self.main(x)
    

class Discriminator(nn.Module):
    def __init__(self, ndf=128, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1),
            # not applying batchnorm to the discriminator input layer
            nn.LeakyReLU(0.2, inplace=True), # the slope of the leak is set to 0.2
            nn.Conv2d(in_channels=ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf*2, out_channels=ndf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf*4, out_channels=ndf*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
            nn.Conv2d(in_channels=ndf*8, out_channels=1, kernel_size=6),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)