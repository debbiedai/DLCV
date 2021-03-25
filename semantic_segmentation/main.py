import torch
import torch.nn as nn
from dataset import Newdataset,SegDataset
from hw2_1 import model_vgg16
import argparse
from torch.utils.data import DataLoader
import numpy as np
# from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
# from mean_iou_evaluate import *
from PIL import Image
# import csv
import pandas as pd 
from os.path import join
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')
parser.add_argument('-ep', '--epoch', default=36, type=int, help='epoch')
parser.add_argument('-lr', '--lr', default=1e-3, help='learning rate')
parser.add_argument('-task', '--task', help='1 or 2', type=int)
parser.add_argument('-mode', '--mode', default='train', type=str, help='test true or false')
parser.add_argument('-resume', '-resume', default=None, help='path to pth', type=str)
parser.add_argument('-g', '--labels', help='ground truth masks directory', type=str)
parser.add_argument('-p', '--pred', help='prediction masks directory', type=str)
parser.add_argument('-test_path','--test_path', type=str, help='testing images directory')
parser.add_argument('-save_path' ,'--save_path', type=str, help='output directory')
parser.add_argument('-improve', '--improve', action="store_true", help='improve true or false')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'



def train(dataloader, net, optimizer, criterion, epoch):
    net.train()
    total = 0
    for index, batch in enumerate(dataloader):
        optimizer.zero_grad()
        img, label, name = batch
        img, label = img.to(device), label.to(device)
        output = net(img)
        loss = criterion(output, label)
        total += loss.item()
        loss.backward()
        optimizer.step()     
        
    print('train loss', total/len(dataloader))


def val(dataloader, net, optimizer, criterion, epoch):
    net.eval()
    total = 0
    correct = 0
    data_num = 0
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            img, label, name = batch
            img, label = img.to(device), label.to(device)
            num = label.size(0)
            output = net(img) 
            pred = torch.argmax(output, axis=1)

            if args.task == 1: 
                correct += (pred==label).sum()
                data_num += label.size(0)          

            if args.task == 2:
                pred = pred.squeeze(1)
                pred_ = pred.cpu().numpy()
                save = np.ones((num,512,512,3))*6
                dict_ = {0:[0,255,255], 1:[255,255,0], 2:[255,0,255], 3:[0,255,0], 4:[0,0,255], 5:[255,255,255], 6:[0,0,0]}
                for p in dict_:
                    b, i, j = np.where(pred_ == p)
                    save[b,i,j,:] = dict_[p]
                for i in range(num):                
                    img_save = save[i,:,:,:]
                    img_save = Image.fromarray(img_save.astype(np.uint8))
                    img_save.save('./p2_data/pred_image/{:s}.png'.format(name[i]))        

            loss = criterion(output, label)
            total += loss.item()
    valid_loss = total/len(dataloader)
    
    print('valid_loss:', valid_loss)

    if args.task == 1:
        print('correct number:', correct.item())
        print('accuracy:', correct.item()/data_num)

    if args.task == 2:
        pred = read_masks(args.pred)
        labels = read_masks(args.labels)
        mean_iou_score(pred, labels)

    return valid_loss
            
def test(dataloader, net, test_path, save_path):
    net.eval()
    
    keep = {'image_id':[], 'label':[]}
    all_ =[]  
    for index, batch in enumerate(dataloader):
        img, name = batch
        img = img.to(device)
        num = img.size(0)
        output = net(img)
        pred = torch.argmax(output, axis=1)

        if args.task == 2:
            pred = pred.squeeze(1)
            pred_ = pred.cpu().numpy()
            save = np.zeros((num,512,512,3))
            dict_ = {0:[0,255,255], 1:[255,255,0], 2:[255,0,255], 3:[0,255,0], 4:[0,0,255], 5:[255,255,255], 6:[0,0,0]}
            for p in dict_:
                b, i, j = np.where(pred_ == p)
                save[b,i,j,:] = dict_[p]
            for i in range(num):                
                img_save = save[i,:,:,:]
                img_save = Image.fromarray(img_save.astype(np.uint8))
                img_save.save(join(args.save_path, '{:s}_mask.png'.format(name[i].split('_')[0])))
                # img_save.save(args.save_path + '{:s}.png'.format(name[i]))
                # img_save.save('./p2_data/pred_image/{:s}.png'.format(name[i]))   
        if args.task == 1:
            for n, p in zip(name, pred):
                all_.append((n, p.item()))


    if args.task == 1:
        all_.sort(key=lambda x: int(x[0].split('_')[0]))
        image_id, label = zip(*all_)
        keep['image_id'] = image_id
        keep['label'] = label
        save_csv = pd.DataFrame(keep)
        save_csv.to_csv(join(args.save_path, 'test_pred.csv'), index=False)  



def main():
    BEST_LOSS = float('inf')
    BATCH_SIZE = args.batch_size   # get batch size from args
    use_pretrained = True

    # task 1
    if args.task == 1:
        if args.mode != 'test':
            train_loader = Newdataset('./p1_data/train_50/',mode='train')
            val_loader = Newdataset('./p1_data/val_50/', mode='valid')
        else:
            test_loader = Newdataset(args.test_path, mode='test')
        model = model_vgg16(pretrained=use_pretrained)
    # task 2
    elif args.task == 2:
        if args.mode != 'test':
            train_loader = SegDataset('./p2_data/train', mode='train')
            val_loader = SegDataset('./p2_data/validation',mode='valid')
        else:
            test_loader = SegDataset(args.test_path, mode='test')
        
        if args.improve:
            from model_improve import model_segmentation
        else:
            from model_baseline import model_segmentation


        model = model_segmentation(pretrained=use_pretrained)
    
    if args.mode != 'test':
        train_loader = DataLoader(train_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_loader, batch_size=BATCH_SIZE)
    else:
        test_loader = DataLoader(test_loader, batch_size=BATCH_SIZE)
    
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model.to(device)
        
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of params = ", total_params)

    if args.mode == 'test':
        # test
        test(test_loader, model, args.test_path, args.save_path)
        return 

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)
    # train
    if args.mode == 'train':
        for epoch in range(args.epoch):
            print('epoch:',epoch)
            train(train_loader, model, optimizer, criterion, args.epoch)
            # val()
            val_loss = val(val_loader, model, optimizer, criterion, args.epoch) ## get the validation loss
            if args.task == 2:
                torch.save(model.state_dict(), './p2_model/p2_%d.pth'%epoch)
            if BEST_LOSS > val_loss:
                torch.save(model.state_dict(), './p{}_best.pth'.format(args.task))
                BEST_LOSS = val_loss


    elif args.mode == 'val':
        # val()
        val_loss = val(val_loader, model, optimizer, criterion, args.epoch)

if __name__ == '__main__':
    main()