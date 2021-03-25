import torch 
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from torch.utils.data import DataLoader
import torch.optim as optim
from GAN import Generator, Discriminator
from dataset import FaceData
from os.path import join 
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt

paresr = argparse.ArgumentParser()
paresr.add_argument('-ep', '--epoch', default=100, type=int, help='epoch')
paresr.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')
paresr.add_argument('-mode', '--mode', default='test', type=str, help='mode')
paresr.add_argument('-save_path', '--save_path', type=str, help='output directory')
paresr.add_argument('-resume', '--resume', type=str, help='resume model path')
args = paresr.parse_args()


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

def loss_function(pred, target):
    criterion = nn.BCELoss().to(device)
    loss = criterion(pred, target)
    # loss = torch.sum(loss)
    return loss

def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, epoch):
    total = 0
    g_total = 0
    d_total = 0
    for i, batch in enumerate(dataloader):
        img, name = batch
        img = img.to(device)
        size = img.size(0)
        # train generator
        optimizer_g.zero_grad()
        noise = torch.empty((size,128)).normal_().to(device)
        gen_img = generator(noise)
        target_one = torch.ones((size,1)).to(device)    # represent real images
        target_zero = torch.zeros((size,1)).to(device)  # fake images
        g_loss = loss_function(discriminator(gen_img), target_one)  # Loss measures generator's ability to fool the discriminator
        g_total += g_loss.item()
        g_loss.backward()
        optimizer_g.step()
        # train disciminator
        optimizer_d.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        
        real_loss = loss_function(discriminator(img), target_one)
        fake_loss = loss_function(discriminator(gen_img.detach()), target_zero)
        d_loss = (real_loss + fake_loss)/2
        d_total += d_loss.item()
        d_loss.backward()
        optimizer_d.step()

        if i == 0:
            torch.manual_seed(0)
            noise = torch.empty((32,128)).normal_().to(device)
            # for j in range(size):
            #     output_save = gen_img[j,:]                
            #     output_save = output_save.view(3,64,64)
            #     output_save = output_save.detach().cpu()
            #     save_image(output_save, join(args.save_path, '{:s}.jpg'.format('fig2_'+str(j))))
            gen_img = generator(noise)
            output_save = gen_img
            output_save = output_save.detach().cpu()
            save_image(output_save, join(args.save_path, '{:s}.jpg'.format('fig2_'+str(epoch))))
        # if i == 1:
        #     gen_img = gen_img.view(size,3,64,64)
        #     output_save = gen_img
        #     output_save = output_save.detach().cpu()
        #     save_image(output_save, join(args.save_path, '{:s}.jpg'.format('fig2_'+str(epoch)+'_1')))


    print('g loss:', g_total/len(dataloader))
    print('d loss:', d_total/len(dataloader))

def test(generator):
    generator.eval()
    torch.manual_seed(0)
    noise = torch.empty((32,128)).normal_().to(device)

    # seed = 16
    # np.random.seed(16)
    # noise = np.random.normal(size=(32,128))
    # noise = torch.from_numpy(noise).cuda().float()
    gen_img = generator(noise)
    gen_img = gen_img.view(32,3,64,64)
    output_save = gen_img
    output_save = output_save.detach().cpu()
    save_image(output_save, join(args.save_path, 'fig2_2.jpg'))



def main():
    best_loss = float('inf')
    batch_size = args.batch_size

    # train_data = FaceData('./hw3_data/face/train/', mode='vae')
    # val_data = FaceData('./hw3_data/face/test/', mode='vae')
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    generator = Generator()
    discriminator = Discriminator()
    if args.resume:
        generator.load_state_dict(torch.load(args.resume))
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    if args.mode == 'test':
        test(generator)
    if args.mode == 'train':
        for epoch in range(args.epoch):
            print('epoch', epoch)
            train(train_loader, generator, discriminator, optimizer_G, optimizer_D, epoch)
            # val_loss = val(val_loader, model, optimizer, args.epoch)
            torch.save(generator.state_dict(), join('./p2_model','g_{:d}.pth'.format(epoch)))
            torch.save(discriminator.state_dict(), join('./p2_model','d_{:d}.pth'.format(epoch)))

if __name__ == '__main__':
    main()