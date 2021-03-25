from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from os import listdir
from torchvision import transforms
import torch 
class Newdataset(Dataset):
    def __init__(self, img_path, mode):
        self.img_path = img_path  
        self.img_name = sorted(os.listdir(self.img_path))
        self.transform1 = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0,0,0),(1,1,1))])
        self.mode = mode
    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, index):
        data = self.img_name[index]
        img_ = os.path.join(self.img_path, self.img_name[index])
        # print(img_)
        img_read = Image.open(img_)
        img = self.transform1(img_read)        
        label = int(data.split('_')[0])        
        if self.mode == 'test':
            return img, data
        else:
            return img, label, data

class SegDataset(Dataset):
    def __init__(self, img_path, mode):
        self.img_path = img_path
        self.mode = mode
        self.all = os.listdir(self.img_path)
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.mask = []
        self.sat = []
        for i in self.all:
            if 'mask' in i:
                self.mask.append(i)
            else:
                self.sat.append(i)
        self.dict_ = {0:[0,255,255], 1:[255,255,0], 2:[255,0,255], 3:[0,255,0], 4:[0,0,255], 5:[255,255,255], 6:[0,0,0]}
    def __len__(self):
        return len(self.all)//2

    def __getitem__(self, index):
        name = self.sat[index].split('.')[0]        
        s = os.path.join(self.img_path, self.sat[index])
        m = os.path.join(self.img_path, self.mask[index])
       
        img = Image.open(s)
        img = np.array(img)
        img_mask = Image.open(m)
        img_mask = np.array(img_mask)
        label = np.ones((512,512)) * 6
        for p in self.dict_:
            i,j = np.where(np.all(img_mask==self.dict_[p], axis=2))
            label[i,j] = p
        img = self.transform1(img)
        label = torch.LongTensor(label)
        # label = self.transform1(label).long()
        if self.mode == 'test':
            return img, name
        else:
            return img, label, name




if __name__ == '__main__':
    # y = Newdataset('./p1_data/val_50/',mode='test')
    z = SegDataset('./p2_data/train/',mode='val')
    # print(y[-1])
    print(z[1])
    