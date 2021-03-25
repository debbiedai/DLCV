import torch 
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from torch.utils.data import DataLoader
import torch.optim as optim
from VAE import VAE
from dataset import FaceData
from os.path import join 
import os
import random
from torchvision.utils import save_image
import matplotlib.pyplot as plt
paresr = argparse.ArgumentParser()
paresr.add_argument('-ep', '--epoch', default=37, type=int, help='epoch')
paresr.add_argument('-mode', '--mode', default='train', type=str, help='mode')
paresr.add_argument('-save_path', '--save_path', type=str, help='output directory')
paresr.add_argument('-resume', '--resume', type=str, help='resume model path')
args = paresr.parse_args()


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

def plot(epoch, mse, kl):
    import matplotlib.pyplot as plt
    x = [i for i in range((epoch+1)*1250)]
    y = mse
    z = kl
    plt.axis([0,40000,0,2000])
    plt.plot(x,z)
    # plt.set_xlim((0, 1.5)
    plt.xlabel("time step")
    plt.ylabel('KLD')
    plt.show()

    # plt.plot(x,z)
    # plt.xlabel("time step")
    # plt.ylabel('KL')
    # plt.show()


reconstruction_function = nn.MSELoss(reduction='mean')
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: original images
    mu: latent mean
    logvar: latent log variance    """

    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KL divergence kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + (1e-5)*KLD, BCE, KLD

mse_list = []
kl_list = []
time_step = []
itera = 0
def train(dataloader, net, optimizer, epoch):
    
    net.train()
    total = 0
    total_mse = 0
    total_kl = 0
    global itera
    
    for i, batch in enumerate(dataloader):        
        itera += i
        time_step.append(i)       
        optimizer.zero_grad()
        img, name = batch
        batch_size = img.size(0)
        img = img.to(device)
        output, mu, logvar, z = net(img,'train')
        img = img.view((batch_size, -1))              
        loss, mse, kl = loss_function(output, img, mu, logvar)
        mse_list.append(mse.item())
        kl_list.append(kl.item())
        total += loss.item()
        total_mse += mse.item()
        total_kl += kl.item()
        loss.backward()
        optimizer.step()
        # if i == 0:
        #     print(name)
        #     for j in range(batch_size):
        #         output_save = output[j,:]
        #         output_save = output_save.view(3,64,64)
        #         # output_save = torch.tensor.numpy(output_save)
        #         output_save = output_save.detach().cpu()
        #         save_image(output_save, join(args.save_path, '{:s}.jpg'.format('fig1_'+str(j))))
        #         # output_save = output_save*255
        #         # output_save = Image.fromarray((output_save).astype(np.uint8))                
        #         # output_save.save(join(args.save_path, '{:s}.jpg'.format('fig1_'+str(j))))
        #         # os.makedirs(args.save_path, exist_ok=True)
    print('train loss:', total/len(dataloader))
    print('mse:', total_mse/len(dataloader))
    print('kl:', total_kl/len(dataloader))
    return time_step, mse_list, kl_list

def val(dataloader, net, optimizer, epoch):
    net.eval()
    total = 0
    total_mse = 0
    total_kl = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img, name = batch
            img = img.to(device)
            num = img.size(0)
            output, mu, logvar, z = net(img, 'val')
            img = img.view(num, -1)
            loss, mse, kl = loss_function(output, img, mu, logvar)
            total += loss.item()
            total_mse += mse.item()
            total_kl += kl.item()
            if i == 0:
                output_save = output
                output_save = output_save.view(num,3,64,64)
                output_save = output_save.detach().cpu()
                # save_image(output_save, join(args.save_path, '{:s}.jpg'.format('fig1_'+str(j))))
                # # print(name)
                # for j in range(num):
                #     # output_save = output[j,:]
                #     # output_save = output_save.view(3,64,64)
                #     # output_save = output_save.permute((1,2,0))
                #     # output_save = output_save.detach().cpu().numpy()
                #     # output_save = output_save*255
                #     # output_save = Image.fromarray(output_save.astype(np.uint8))                
                #     # output_save.save(join(args.save_path, '{:s}.jpg'.format('fig1_'+str(j))))
                #     output_save = output[j,:]
                #     output_save = output_save.view(3,64,64)
                #     output_save = output_save.detach().cpu()
                #     save_image(output_save, join(args.save_path, '{:s}.jpg'.format('fig1_'+str(j))))

        val_loss = total/num
        print('val_loss', val_loss)
        print('mse:', total_mse/len(dataloader))
        print('kl:', total_kl/len(dataloader))
    return val_loss
            
def test(net):
    net.eval()

    seed = 16
    np.random.seed(16)
    noise = np.random.normal(loc=0, scale=1, size=(32,512))
    noise = torch.from_numpy(noise).cuda().float()
    save = torch.ones(32,3,64,64)
    for i in range(32):
        img = noise[i][:512].reshape((1,512))
        output = net(img, 'test')
        output_save = output.view(3,64,64)
        save[i,:,:,:] = output_save
        output_save = output_save.detach().cpu()
    save_image(save, join(args.save_path, 'fig1_4.jpg'))


    # seed = [2,4,6,11,14,16,20,24,26,30]
    # for i in seed:
    #     np.random.seed(i)
    #     noise = np.random.normal(loc=0, scale=1, size=(1,512))
    #     noise = torch.from_numpy(noise).cuda().float()
    #     output = net(noise, 'test')
    #     output_save = output.view(3,64,64)
    #     output_save = output_save.detach().cpu()
    #     save_image(output_save, join(args.save_path, '{:s}.jpg'.format('fig1_'+str(i))))


            # gen_img = gen_img.view(size,3,64,64)
            # output_save = gen_img
            # output_save = output_save.detach().cpu()
            # save_image(output_save, join(args.save_path, '{:s}.jpg'.format('fig2_'+str(epoch)+'_1')))
def main():
    best_loss = float('inf')
    batch_size = 32
    # train_data = FaceData('./hw3_data/face/train/', mode='vae')
    # val_data = FaceData('./hw3_data/face/test/', mode='vae')
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    model = VAE()
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if args.mode == 'test':
        test(model)
    if args.mode == 'train':

        for epoch in range(args.epoch):
            print('epoch', epoch)

            time_step, mse_list, kl_list = train(train_loader, model, optimizer, args.epoch)
            val_loss = val(val_loader, model, optimizer, args.epoch)
            # torch.save(model.state_dict(), join('./p1_model','{:d}.pth'.format(epoch)))
        # if val_loss < best_loss:
        #     torch.save(model.state_dict(), join('./p1_model', 'best_{:d}.pth'.format(epoch)))
        #     best_loss = val_loss
            # print(time_step)
            # print(mse_list)
            # print(kl_list)
            if epoch == 35:
                plot(epoch, mse_list, kl_list)
    elif args.mode == 'val':
        val_loss = val(val_loader, model, optimizer, args.epoch)

            
if __name__ == '__main__':
    main()