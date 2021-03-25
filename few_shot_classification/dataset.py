import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def filenameToPILImage(x):
    return Image.open(x)
# filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.labels = self.data_df['label'].tolist()
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        

    def __getitem__(self, index):
        # print(index)
        images = []
        all_labels = []
        label2num = []
        for i in index:
            path = self.data_df.loc[i.item(), "filename"]
            label = self.data_df.loc[i.item(), "label"]

            image = self.transform(os.path.join(self.data_dir, path))
            images.append(image)
            all_labels.append(label)

        label_name = set(all_labels)
        length = len(label_name)
        num = [i for i in range(length)]
        dict_ = dict(zip(label_name, num))
        for x in all_labels:
            label2num.append(dict_[x])

        label2num = torch.tensor(label2num)
        images = torch.stack(images)
        return images, label2num

    def __len__(self):
        return len(self.data_df)

class CategoriesSampler(Sampler):
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        classes = np.unique(label)
        length = len(classes)
        idx = [i for i in range(length)]
        dict_ = dict(zip(classes,idx))
        label2num = []
        for i in label:
            label2num.append(dict_[i])
        label = np.array(label2num)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch



def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--train_way', default=5, type=int, help='N_way (default:5)')

    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    test_dataset = MiniDataset('./hw4_data/val.csv','./hw4_data/val')
    labels = test_dataset.labels
    # print(test_dataset)
    sampler = CategoriesSampler(labels, 5, args.train_way, args.N_query + args.N_shot)
    # batch_sample = BatchSampler(test_dataset, 2,2,1)
    # print(batch_sample)
    # batch = BatchSampler(label, )
    test_loader = DataLoader(
        test_dataset,
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=sampler)
    for j in range(1):
        for i , batch in enumerate(test_loader):
            img, label_ = batch
            # print(img.size())
            if i==1:
                break


    