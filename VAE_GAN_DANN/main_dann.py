import torch 
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from torch.utils.data import DataLoader
import torch.optim as optim
from DANN import Extractor, Class_classifier, Domain_classifier
from dataset import DigitData
from os.path import join 
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.autograd import Function
import pandas as pd
paresr = argparse.ArgumentParser()
paresr.add_argument('-ep', '--epoch', default=85, type=int, help='epoch')
paresr.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')
paresr.add_argument('-data_path', '--data_path', type=str, help='directory of testing images')
paresr.add_argument('-target', '--target', type=str, help='name of target domain')
paresr.add_argument('-mode', '--mode', default='train', type=str, help='mode')
paresr.add_argument('-t_mode', '--training_mode', default='dann', type=str, help='training mode')
paresr.add_argument('-save_path', '--save_path', type=str, help='output directory')
paresr.add_argument('-gamma', '--gamma', default=10, type=int, help='gamma')
args = paresr.parse_args()


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

def train(training_mode, source_dataloader, target_dataloader, extractor, class_classifier, domain_classifier, c_criterion, d_criterion, optimizer, epoch):
    extractor.train()
    class_classifier.train()
    domain_classifier.train()
    total = 0
    total_classloss = 0
    total_domainloss = 0
    # steps
    start_steps = epoch * len(source_dataloader)
    total_steps = args.epoch * len(source_dataloader)
    data_len = len(source_dataloader)
    # print(len(source_dataloader), len(target_dataloader)) #represent how many batchs
    for i, (s_data, t_data) in enumerate(tqdm(zip(source_dataloader, target_dataloader), ncols=50)):
        if training_mode == 'dann':
            p = float(i + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-args.gamma * p)) - 1

            img1, label1, name1 = s_data
            img2, label2, name2 = t_data
            size = min((img1.shape[0], img2.shape[0]))
            img1, label1 = img1[0:size, :, :, :], label1[0:size]
            img2, label2 = img2[0:size, :, :, :], label2[0:size]
            img1, label1, img2, label2 = img1.to(device), label1.to(device), img2.to(device), label2.to(device)
            size1 = img1.size(0)
            size2 = img2.size(0)
            optimizer.zero_grad()
            source_labels = torch.zeros(size1).long().to(device)
            target_labels = torch.ones(size2).long().to(device)

            s_feature = extractor(img1)
            t_feature = extractor(img2)
            class_pred = class_classifier(s_feature)
            class_loss = c_criterion(class_pred, label1)
            total_classloss += class_loss.item()
            t_pred = domain_classifier(t_feature, constant)
            s_pred = domain_classifier(s_feature, constant)
            t_loss = d_criterion(t_pred, target_labels)
            s_loss = d_criterion(s_pred, source_labels)
            domain_loss = (t_loss+s_loss)
            total_domainloss += domain_loss.item()

            loss = class_loss + domain_loss
            total += loss.item()
            loss.backward()
            optimizer.step()

        elif training_mode == 'source':
            img1, label1, name1 = s_data
            img1, label1 = img1.to(device), label1.to(device)
            size = img1.size(0)

            optimizer = optim.SGD(list(extractor.parameters())+list(class_classifier.parameters()), lr=1e-3, momentum=0.9)
           
            optimizer.zero_grad()

            s_feature = extractor(img1)

            class_pred = class_classifier(s_feature)
            class_loss = c_criterion(class_pred, label1)
            total += class_loss.item()
            class_loss.backward()
            optimizer.step()

            # if (i+1)%10 == 0:
            #     print('Class Loss: {:.6f}'.format(class_loss.item()))

        elif training_mode == 'target':
            img2, label2, name2 = t_data
            img2, label2 = img2.to(device), label2.to(device)
            size = img2.size(0)

            optimizer = optim.SGD(list(extractor.parameters()) + list(class_classifier.parameters()), lr=1e-3, momentum=0.9)
            optimizer.zero_grad()

            t_feature = extractor(img2)

            class_pred = class_classifier(t_feature)
            class_loss = c_criterion(class_pred, label2)

            class_loss.backward()
            optimizer.step()

            if (i+1)%10 == 0:
                print('Loss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(loss.item(), class_loss.item(), domain_loss.iten()))
    if training_mode == 'dann':    
        print('Loss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(total/len(source_dataloader), total_classloss/len(source_dataloader), total_domainloss/len(source_dataloader)))
    elif training_mode == 'source':
        print('train loss:', total/data_len)
def val(dataloader, extractor, class_classifier, class_criterion, epoch):
    extractor.eval()
    class_classifier.eval()
    total = 0
    total_loss = 0
    correct = 0
    for i, batch in enumerate(tqdm(dataloader, ncols=50)):
        img, label, name = batch
        size = img.size(0)
        img, label = img.to(device), label.to(device)
        feature = extractor(img)
        class_pred = class_classifier(feature)
        pred = torch.argmax(class_pred, axis=1)
        correct += ((pred==label).sum()).item()
        total += size
        loss = class_criterion(class_pred, label)
        total_loss += loss.item()  
    val_acc = correct/total
    print('val loss:', total_loss/len(dataloader))
    print('correct:', correct)
    print('val_accuracy:', val_acc)
    return val_acc

def test(dataloader, extractor, class_classifier):
    extractor.eval()
    class_classifier.eval()
    total = 0
    keep = {'image_name':[], 'pred':[]}
    all_=[]
    for i, batch in enumerate(dataloader):
        img, name = batch
        size = img.size(0)
        img = img.to(device)
        feature = extractor(img)
        class_pred = class_classifier(feature)
        pred = torch.argmax(class_pred, axis=1)
        for n,p in zip(name,pred):
            all_.append((n, p.item()))
        all_.sort(key=lambda x: int(x[0].split('.')[0]))
        name_, pred_ = zip(*all_)
        keep['image_name'] = name_
        keep['pred'] = pred_
        save_csv = pd.DataFrame(keep)
        save_csv.to_csv(join(args.save_path, 'test_pred.csv'), index=False)

def main():
    best_loss = float('inf')
    batch_size = args.batch_size
    # if mode = train or val, csv_path exist
    # ------target data------
    # train_tdata = DigitData('./hw3_data/digits/mnistm/train/', 'mnistm', mode='train', csv_path='./hw3_data/digits/mnistm/train.csv')
    # test_tdata = DigitData('./hw3_data/digits/mnistm/test/', 'mnistm', mode='val', csv_path='./hw3_data/digits/mnistm/test.csv')
    # train_size = int(0.8 * len(train_tdata))
    # val_size = len(train_tdata) - train_size
    # train_tdataset, val_tdataset = torch.utils.data.random_split(train_tdata, [train_size, val_size])

    
    # train_tdataloader = DataLoader(train_tdataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_tdataloader = DataLoader(val_tdataset, batch_size=batch_size, shuffle=False)
    # test_tdataloader = DataLoader(test_tdata, batch_size=batch_size, shuffle=False)

    # #------source data-------
    # train_sdata = DigitData('./hw3_data/digits/usps/train/', 'usps', mode='train', csv_path='./hw3_data/digits/usps/train.csv')
    # test_sdata = DigitData('./hw3_data/digits/usps/test/', 'usps', mode='val', csv_path='./hw3_data/digits/usps/test.csv')
    # train_size = int(0.8 * len(train_sdata))
    # val_size = len(train_sdata) - train_size
    # train_sdataset, val_sdataset = torch.utils.data.random_split(train_sdata, [train_size, val_size])

    
    # train_sdataloader = DataLoader(train_sdataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_sdataloader = DataLoader(val_sdataset, batch_size=batch_size, shuffle=False)
    # test_sdataloader = DataLoader(test_sdata, batch_size=batch_size, shuffle=False)
    
    test_data = DigitData(args.data_path, args.target, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    extractor = Extractor().to(device)
    class_classifier = Class_classifier().to(device)
    domain_classifier = Domain_classifier().to(device)
    
    if args.target == 'usps':
        extractor.load_state_dict(torch.load('./svhn_usps_extractor.pth'))
        class_classifier.load_state_dict(torch.load('./svhn_usps_classifier.pth'))
    elif args.target == 'mnistm':
        extractor.load_state_dict(torch.load('./usps_mnistm_extractor.pth'))
        class_classifier.load_state_dict(torch.load('./usps_mnistm_classifier.pth'))
    elif args.target == 'svhn':
        extractor.load_state_dict(torch.load('./mnistm_svhn_extractor.pth'))
        class_classifier.load_state_dict(torch.load('./mnistm_svhn_classifier.pth'))

    class_criterion = nn.NLLLoss().to(device)
    domain_criterion = nn.NLLLoss().to(device)

    optimizer = optim.Adam([{'params': extractor.parameters()},
                        {'params': class_classifier.parameters()},
                        {'params': domain_classifier.parameters()}], lr=1e-3, weight_decay=0.0001)
    if args.mode == 'test':
        test(test_dataloader, extractor, class_classifier)
    elif args.mode == 'train':
        for epoch in range(args.epoch):
            print('epoch:', epoch)
            train(args.training_mode, train_sdataloader, train_tdataloader, extractor, class_classifier, domain_classifier, class_criterion, domain_criterion, optimizer, epoch)
            val_loss = val(val_sdataloader, extractor, class_classifier, class_criterion, epoch)
            test(test_tdataloader, extractor, class_classifier)
            torch.save(extractor.state_dict(), './p3_model/usps_mnistm/p3_usps_mnistm_{}_extractor.pth'.format(epoch))
            torch.save(class_classifier.state_dict(), './p3_model/usps_mnistm/p3_usps_mnistm_{}_class.pth'.format(epoch))
            torch.save(domain_classifier.state_dict(), './p3_model/usps_mnistm/p3_usps_mnistm_{}_domain.pth'.format(epoch))



if __name__ == '__main__':
    main()
