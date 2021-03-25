import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from os.path import join
import argparse
from torch.utils.data import DataLoader
import pandas as pd
from dataset import MiniDataset, CategoriesSampler
from convnet import Convnet, parametric
from tqdm import tqdm
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description='Few shot learning')
    parser.add_argument('--train_way', default=5, type=int, help='N_way (default:5)')
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default:5)')
    parser.add_argument('--N-shot', default=10, type=int, help='N_shot (default:1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default:15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")
    parser.add_argument('--epoch', type=int, default=50, help='epochs')
    parser.add_argument('--mode', type=str, default='euc', help='distance function')

    return parser.parse_args()

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cosine_similarity(a, b):
    cosine = torch.nn.CosineSimilarity(dim=2)
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n,m,-1)
    b = b.unsqueeze(0).expand(n,m,-1)
    output = cosine(a,b)
    return output



def train(args, dataloader, net, parametric_model, optimizer, optimizer_para, epoch):
    net.train()
    criterion = nn.CrossEntropyLoss().to(device)
    correct = 0
    total = 0
    total_loss = 0
    total_acc = 0
    for i, (img, label) in enumerate(tqdm(dataloader, ncols=50)):
        optimizer.zero_grad()
        img, label = img.to(device), label.to(device)
        img = img.squeeze()
        # total += img.size(0)
        p = args.N_shot * args.train_way
        img_shot, img_query = img[:p], img[p:]
        proto = net(img_shot)   #(5,1600)
        proto = proto.reshape(args.N_shot, args.train_way, -1).mean(dim=0)
        label_ = torch.arange(args.train_way).repeat(args.N_query).long().to(device)
        # logits = euclidean_metric(net(img_query), proto)    #(75,5)
        logits = cosine_similarity(net(img_query), proto)
        # logits = parametric_model(net(img_query), proto)
        
        loss = criterion(logits, label_)
        total_loss += loss.item()
        acc = count_acc(logits, label_)
        total_acc += acc
  
        loss.backward()
        optimizer.step()
        if args.mode == 'parametric':
            optimizer_para.step()
    print('episode:', epoch)
    print('train loss:{:4f} acc:{:4f}'.format(total_loss/len(dataloader), total_acc/len(dataloader)))

              

def val(args, dataloader, net, epoch):
    net.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    total_loss = 0
    total_acc = 0
    for i, (img, label) in enumerate(tqdm(dataloader, ncols=50)):
        img, label = img.to(device), label.to(device)
        img = img.squeeze()
        p = args.N_shot * args.N_way
        img_shot, img_query = img[:p], img[p:]

        proto = net(img_shot)   #(5,1600)
        proto = proto.reshape(args.N_shot, args.N_way, -1).mean(dim=0)

        label_ = torch.arange(args.N_way).repeat(args.N_query).long().to(device)
        # logits = euclidean_metric(model(img_query), proto)
        logits = cosine_similarity(net(img_query), proto)
        # logits = parametric_model(net(img_query), proto)


        loss = criterion(logits, label_)
        total_loss += loss.item()
        acc = count_acc(logits, label_)
        total_acc += acc
    val_acc = total_acc/len(dataloader)
    print('val loss:{:4f} acc:{:4f}'.format(total_loss/len(dataloader), val_acc))
    return val_acc









def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
if __name__=='__main__':
    args = parse_args()
    data_train = MiniDataset('./hw4_data/train.csv','./hw4_data/train')
    labels_train = data_train.labels
    sampler_train = CategoriesSampler(labels_train, 2400, args.train_way, args.N_query + args.N_shot)
    dataloader_train = DataLoader(data_train, num_workers=4,
        pin_memory=False, worker_init_fn=worker_init_fn, sampler=sampler_train)
    data_val = MiniDataset('./hw4_data/val.csv','./hw4_data/val')
    labels_val = data_val.labels
    sampler_val = CategoriesSampler(labels_val, 600, args.N_way, args.N_query + args.N_shot)
    dataloader_val = DataLoader(data_val,
        pin_memory=False, worker_init_fn=worker_init_fn, sampler=sampler_val)

    model = Convnet().to(device)
    parametric_model = parametric().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer_para = optim.Adam(model.parameters(), lr=1e-4)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for epoch in range(args.epoch):    
        train(args, dataloader_train, model, parametric_model, optimizer, optimizer_para, epoch)
        val_acc = val(args, dataloader_val, model, epoch)
        torch.save(model.state_dict(), './p1_model/5way1shot_cosine/epoch_{}_{:.4f}.pth'.format(epoch, val_acc))
        if args.mode == 'parametric':
            torch.save(parametric_model.state_dict(), './p1_model/5way1shot_cosine/epoch_{}_para.pth'.format(epoch))

    