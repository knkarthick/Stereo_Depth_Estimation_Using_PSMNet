from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
from torch.utils.data.sampler import SubsetRandomSampler 
import json
from collections import defaultdict
# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

#to obtain training indices that will be used for validation
valid_size=0.2
num_train = len(all_left_img)
print ("Len of all_left_img: ", str(num_train))
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size*num_train))
train_idx, valid_idx = indices[split:], indices[:split]

#define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
#print (train_idx)
print ('***********')
##print (valid_idx)

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
  
    if (a_set & b_set):
        print(a_set & b_set)
    else:
        print("No common elements")

print (common_member(train_idx, valid_idx))


print (len(train_sampler), len(valid_sampler))




TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 4, shuffle= False, sampler =train_sampler, num_workers= 4, drop_last=False)

ValidationImgLoader = torch.utils.data.DataLoader(DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), #had to make shuffle False
         batch_size= 4, shuffle= False, sampler =valid_sampler ,num_workers= 4, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 4, shuffle= False, num_workers= 4, drop_last=False)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
print(torch.cuda.is_available())


optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def train(imgL,imgR, disp_L):
        model.train()

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output,1)
            loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data

def validation(imgL,imgR, disp_L):
    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    #---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    #----
    
    if args.model == 'stackhourglass':
        output3 = model(imgL,imgR)
        #output1 = torch.squeeze(output1,1)
        #output2 = torch.squeeze(output2,1)
        output3 = torch.squeeze(output3,1)
        loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
        #loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
    elif args.model == 'basic':
        output = model(imgL,imgR)
        output = torch.squeeze(output,1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    return loss.data

def test(imgL,imgR,disp_true):
    #print ("In test")

    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    #---------
    mask = disp_true < 192
    #----

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16       
        top_pad = (times+1)*16 -imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3]//16                       
        right_pad = (times+1)*16-imgL.shape[3]
    else:
        right_pad = 0  

    imgL = F.pad(imgL,(0,right_pad, top_pad,0))
    imgR = F.pad(imgR,(0,right_pad, top_pad,0))

    with torch.no_grad():
        output3 = model(imgL,imgR) #modified
        output3 = torch.squeeze(output3)
    
    if top_pad !=0:
        img = output3[:,top_pad:,:]
    else:
        img = output3

    if len(disp_true[mask])==0:
        loss = 0
    else:
        loss = F.l1_loss(img[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

    return loss.data.cpu()

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    valid_loss_min = np.Inf # set initial "min" to infinity
    train_loss_dic = {}
    train_time_dic = {}
    valid_loss_dic = {}
    valid_time_dic = {}
    test_loss_dic = {}
    start_full_time = time.time()
    
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        total_train_time = 0
        total_validation_time = 0
        total_validation_loss = 0
        adjust_learning_rate(optimizer,epoch)
        torch.cuda.empty_cache()
        
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
            total_train_time += time.time() - start_time
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        print ("Total Train Time: ", str(total_train_time))
        
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)), file=open("resnet_output_train_sceneflow.txt", "a"))
        print ("Total Train Time: ", str(total_train_time), file=open("resnet_output_train_sceneflow.txt", "a"))
        #train_loss_dic[epoch] = [total_train_loss/len(TrainImgLoader).item(), total_train_time]
        train_loss_dic[epoch] = total_train_loss/len(TrainImgLoader)
        train_time_dic[epoch] = total_train_time
        print (train_loss_dic)

        print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

        ## validation ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(ValidationImgLoader):
            start_time_validation = time.time()
            valid_loss = validation(imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d validation loss = %.3f , time = %.2f' %(batch_idx, valid_loss, time.time() - start_time_validation)) 
            # update running validation loss 
            total_validation_loss += valid_loss
            total_validation_time += time.time() - start_time_validation
        print('epoch %d total validation loss = %.3f' %(epoch, total_validation_loss/len(ValidationImgLoader)))
        print ("Total Validation Time: ", str(total_validation_time))
        print('epoch %d total validation loss = %.3f' %(epoch, total_validation_loss/len(ValidationImgLoader)), file=open("resnet_output_val_sceneflow.txt", "a"))
        print ("Total Validation Time: ", str(total_validation_time), file=open("resnet_output_val_sceneflow.txt", "a"))
        #valid_loss_dic[epoch] = [total_validation_loss/len(ValidationImgLoader).item(), total_validation_time]
        valid_loss_dic[epoch] = total_validation_loss/len(ValidationImgLoader)
        valid_time_dic[epoch] = total_validation_time
        print (valid_loss_dic)
        

        #train_loss_dic = {k: v.item() for k, v in train_loss_dic.items()}
        #valid_loss_dic = {k: v.item() for k, v in valid_loss_dic.items()}

        #SAVE
        savefilename = args.savemodel+'/resentcheckpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader),
		}, savefilename)  

    
    
    print ("Writing train_loss_dic to json file...")
    train_loss_dic = {k: v.item() for k, v in train_loss_dic.items()}
    print (train_loss_dic)
    final_train_dic = defaultdict(list)
    for d in (train_loss_dic, train_time_dic):
        for key, value in d.items():
            final_train_dic[key].append(value)
    
    final_train_dic = dict(final_train_dic)

    
    valid_loss_dic = {k: v.item() for k, v in valid_loss_dic.items()}
    print (valid_loss_dic)

    final_validation_dic = defaultdict(list)
    for d in (valid_loss_dic, valid_time_dic):
        for key, value in d.items():
            final_validation_dic[key].append(value)
    
    final_validation_dic = dict(final_validation_dic)
    final_dict = {}
    final_dict['training'] = final_train_dic
    final_dict['validation'] = final_validation_dic

    json_object_temp = json.dumps(final_dict, indent = 4)
    with open("train_val_intermediary.json", "a") as outfile:
        outfile.write(json_object_temp)
	#------------- TEST ------------------------------------------------------------
    
    total_test_loss = 0
    total_test_time = 0
    test_time_dic = {}
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        start_time_test = time.time()
        test_loss = test(imgL,imgR, disp_L)
        print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
        total_test_time += time.time() - start_time_test
        total_test_loss += test_loss
    
    print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
    test_loss_dic[epoch] = total_test_loss/len(TestImgLoader)
    test_time_dic[epoch] = total_test_time
    test_loss_dic = {k: v.item() for k, v in test_loss_dic.items()}
    
    final_test_dic = defaultdict(list)
    for d in (test_loss_dic, test_time_dic):
        for key, value in d.items():
            final_test_dic[key].append(value)
    
    final_test_dic = dict(final_test_dic)
    
    final_dict['test'] = final_test_dic



    json_object = json.dumps(final_dict, indent = 4)
    with open("final.json", "w") as outfile:
        outfile.write(json_object)
	#----------------------------------------------------------------------------------
	#SAVE test information
    
    savefilename = args.savemodel+'testinformation.tar'
    torch.save({
		    'test_loss': total_test_loss/len(TestImgLoader),
		}, savefilename)


if __name__ == '__main__':
   main()
    
