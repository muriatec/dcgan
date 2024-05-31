import argparse
import torch
import torchvision
from tqdm import tqdm
from dcgan import Generator, Discriminator,init_weights
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import STL10

def main(args):
    if args.dataset=="stl-10":
        train_dataset=STL10('./dataset',split='unlabeled',download=True, transform=transforms.Compose(
            [transforms.ToTensor()]
        ))
        trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2)
    generator=Generator()
    discriminator=Discriminator()
    adversarial_loss = torch.nn.BCELoss()
    
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # generator.train()
    # discriminator.train()
    for epoch in range(args.epochs):
        for item in tqdm(trainloader):
            # print(item)
            input=item[0]
            input=input.to('cuda')
            valid = torch.full((input.shape[0],1), 1.0,dtype=torch.float,device='cuda')
            fake = torch.full((input.shape[0], 1), 0.0,dtype=torch.float,device='cuda')
            optimizer_G.zero_grad()
            z = torch.randn((input.shape[0], 100,1,1),device='cuda')
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            
            real_loss = adversarial_loss(discriminator(input), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"% (epoch, args.epochs, d_loss.item(), g_loss.item()))
            

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()          
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs of training")  
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")          
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")            
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient") 
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient") 
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") 
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")  
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")  
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")      
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")  
    parser.add_argument("--dataset",type=str,default='stl-10',help="dataset used to train the model")
    args = parser.parse_args()
    main(args)
