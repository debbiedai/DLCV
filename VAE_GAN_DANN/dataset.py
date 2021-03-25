import torch
from PIL import Image
import os
from os import listdir
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
class FaceData(Dataset):
    def __init__(self, img_path, mode):
        self.img_path = img_path
        self.mode = mode
        self.all = os.listdir(self.img_path)
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        # self.transfrom2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    def __len__(self):
        return len(self.all)
        

    def __getitem__(self, index):
        data = self.all[index]
        name = self.all[index].split('.')[0]
        data_ = os.path.join(self.img_path, data)
        img_read = Image.open(data_)
        if self.mode == 'gan':
            img = self.transform2(img_read)
        else:
            img = self.transform1(img_read)
        return img, name



class DigitData(Dataset):
  
    def __init__(self, img_path, source, mode, csv_path=None):
        self.img_path = img_path
        self.source = source
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])
        if mode != 'test':
            df = pd.read_csv(csv_path)
            self.label = df['label']
            self.name = df['image_name']
        else:
            self.name = os.listdir(img_path)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        if self.mode != 'test':
            name = self.name[index]
            label = self.label[index]
            img = Image.open(os.path.join(self.img_path, name))
        else:
            name = self.name[index]
            img = Image.open(os.path.join(self.img_path, name))
        img = np.array(img)
        if len(img.shape) != 3:
            img = np.repeat(img.reshape(28, 28, 1), 3, 2)
        img = self.transform(img)
        if self.mode != 'test':
            return img, label, name
        return img, name
            


        # gt = self.label[index]
        # name = self.all[index].split('.')[0]    
        # data = self.all[index]
        # data_ = os.path.join(self.img_path, data)
        # img_read = Image.open(data_)
        # npy = np.array(img_read)
        # img = self.transfrom1(img_read)
        # # if npy.shape[0] != 3:
        # # import matplotlib.pyplot as plt
        # # plt.imshow(np.array(img_read))
        # # plt.show()
        # # print(gt)
        # if self.datatype == 'usps':
        #     # img = np.repeat()
        #     self.usps_img[0,:,:] = img
        #     self.usps_img[1,:,:] = img
        #     self.usps_img[2,:,:] = img
            
        #     if self.mode == 'test':
        #         return self.usps_img
        #     else:
        #         return self.usps_img, gt, name
        
        # if self.mode == 'test':
        #     return img
        # else:
        #     return img, gt, name
if __name__ == '__main__':
    x = DigitData('./hw3_data/digits/usps/train/', datatype = 'usps', mode='train',csv_path='./hw3_data/digits/usps/train.csv')
    train_size = int(0.8 * len(x))
    val_size = len(x) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(x, [train_size, val_size])
    val_sdataloader = DataLoader(val_dataset, batch_size=10, shuffle=True)
    
    for index, (imgs, gts, names) in enumerate(val_sdataloader):

        for i in range(imgs.size(0)):
            import matplotlib.pyplot as plt
            print('FUCK')
            print(gts[i])
            print(names[i])
            tmp = imgs[i]
            tmp = tmp.permute(1, 2, 0).numpy()
            plt.imshow(tmp)
            plt.show()
        

    

    # x = FaceData('./hw3_data/face/train/', mode='gan')
    # y = FaceData('./hw3_data/face/train/', mode='vae')
    # print(x[0])
    # print(y[0])


