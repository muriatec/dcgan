import argparse
import torch
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
from dcgan import Generator, Discriminator, init_weights
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import datetime

def main(args):
    if args.dataset=="stl-10":
        train_dataset=STL10('./dataset', split='unlabeled', download=True, transform=transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator(ngf=args.ngf, img_size=args.img_size)
    discriminator = Discriminator(ndf=args.ndf, img_size=args.img_size)
    loss = torch.nn.BCELoss()
    
    generator.to(device)
    discriminator.to(device)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    G_losses = []
    D_losses = []

    for epoch in range(args.epochs):
        for i, (images, _) in enumerate(trainloader):
            real_images = images.to(device)

            valid = torch.full((real_images.shape[0], 1), 1.0, dtype=torch.float, device=device)
            fake = torch.full((real_images.shape[0], 1), 0.0, dtype=torch.float, device=device)

            # train generator
            optimizer_G.zero_grad()

            # random noise
            z = torch.randn((real_images.shape[0], 100, 1, 1), device=device)
            fake_imgs = generator(z)

            g_loss = loss(discriminator(fake_imgs).view(-1), valid)
            g_loss.backward()
            optimizer_G.step()

            # train discriminator
            optimizer_D.zero_grad()
            
            real_loss = loss(discriminator(real_images).view(-1), valid)
            fake_loss = loss(discriminator(fake_imgs.detach()).view(-1), fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            print("[Epoch %d/%d] [Iteration %d/%d] [D loss: %f] [G loss: %f]" % 
                  (epoch, args.epochs, i, len(trainloader), d_loss.item(), g_loss.item()))

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
    
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    torch.save(generator.state_dict(), './model/{time}_ngf{args.ngf}_ndf{args.ndf}_lr{args.lr}_bs{args.batch_size}_epoch{args.epochs}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()          
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs of training")  
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")          
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")            
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient") 
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient") 
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") 
    parser.add_argument("--ngf", type=int, default=64, help="number of generator features") 
    parser.add_argument("--ndf", type=int, default=64, help="number of discriminator features") 
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")  
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")      
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")  
    parser.add_argument("--dataset",type=str,default='stl-10',help="dataset used to train the model")
    args = parser.parse_args()
    main(args)