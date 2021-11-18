from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import copy
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA
from torch.utils.data.sampler import SubsetRandomSampler 
import json
from collections import defaultdict

from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
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

if args.datatype == '2015':
   from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
   from dataloader import KITTIloader2012 as ls

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

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



# TrainImgLoader = torch.utils.data.DataLoader(
#          DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
#          batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

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
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

def train(imgL,imgR,disp_L):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        #---------
        mask = (disp_true > 0)
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
            output = torch.squeeze(output3,1)
            loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        #return loss.data[0]
        return loss.data.item()

def validation(imgL,imgR, disp_true):
    model.eval()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))   
    
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL,imgR)


    pred_disp = output3.data.cpu()
    pred_disp = pred_disp.squeeze(axis=1)

    
    #computing 3-px error#
    true_disp = copy.deepcopy(disp_true)
    index = np.argwhere(true_disp>0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
    torch.cuda.empty_cache()

    return 1-(float(torch.sum(correct))/float(len(index[0])))


def test(imgL,imgR,disp_true):
    model.eval()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))   
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL,imgR)

    pred_disp = output3.data.cpu()
    pred_disp = pred_disp.squeeze(axis=1)

    #computing 3-px error#
    true_disp = copy.deepcopy(disp_true)
    index = np.argwhere(true_disp>0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
    torch.cuda.empty_cache()

    return 1-(float(torch.sum(correct))/float(len(index[0])))

def adjust_learning_rate(optimizer, epoch):
    #if epoch <= 200:
    if epoch <= 100:
       lr = 0.001
    else:
       lr = 0.0001
    #lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc=0
    max_epo=0
    max_epo_validation = 0
    max_acc_validation = 0
    train_loss_dic = {}
    train_time_dic = {}
    valid_loss_dic = {}
    valid_time_dic = {}
    test_loss_dic = {}
    start_full_time = time.time()

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        total_validation_loss = 0
        total_train_time = 0
        total_validation_time = 0
        adjust_learning_rate(optimizer,epoch)
        torch.cuda.empty_cache()
           
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time() 
            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
            total_train_time += time.time() - start_time
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)), file=open("output_train.txt", "a"))
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        
        print('epoch %d Total Train Time = %.3f' %(epoch, total_train_time))
        print('epoch %d Total Train Time = %.3f' %(epoch, total_train_time), file=open("output_train.txt", "a"))
        
        #print ("epoch %d Total Train Time: ", str(total_train_time))
        train_loss_dic[epoch] = total_train_loss/len(TrainImgLoader)
        train_time_dic[epoch] = total_train_time
        print (train_loss_dic)
        print (train_time_dic)
        #print (train_loss_dic, file=open("output_train.txt", "a"))
        #print (train_time_dic, file=open("output_train.txt", "a"))
        print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
	   
        ## validation ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(ValidationImgLoader):
            start_time_validation = time.time()
            valid_loss = validation(imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d 3-px error in val = %.3f' %(batch_idx, valid_loss*100))
            # update running validation loss 
            total_validation_loss += valid_loss
            total_validation_time += time.time() - start_time_validation
        
        print('epoch %d total 3-px error in val = %.3f' %(epoch, total_validation_loss/len(ValidationImgLoader)*100))
        print ('epoch %d total time in val = %.3f' %(epoch, total_validation_time))
        
        print('epoch %d total 3-px error in val = %.3f' %(epoch, total_validation_loss/len(ValidationImgLoader)*100), file=open("output_validation.txt", "a"))
        print ('epoch %d total time in val = %.3f' %(epoch, total_validation_time), file=open("output_validation.txt", "a"))
        valid_loss_dic[epoch] = total_validation_loss/len(ValidationImgLoader)
        valid_time_dic[epoch] = total_validation_time
        print (valid_loss_dic)
        print (valid_time_dic)
        #print (valid_loss_dic, file=open("output_validation.txt", "a"))
        #print (valid_time_dic, file=open("output_validation.txt", "a"))

        if total_validation_loss/len(ValidationImgLoader)*100 > max_acc:
            max_acc_validation = total_validation_loss/len(ValidationImgLoader)*100
            max_epo_validation = epoch
        print('MAX epoch %d total validation error = %.3f' %(max_epo_validation, max_acc_validation))


        #SAVE
        if epoch % 10 ==0:
            savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss/len(TrainImgLoader),
            }, savefilename) 

    print ("Writing train_loss_dic to json file...")
    #train_loss_dic = {k: v.item() for k, v in train_loss_dic.items()}
    print (train_loss_dic)
    final_train_dic = defaultdict(list)
    for d in (train_loss_dic, train_time_dic):
        for key, value in d.items():
            final_train_dic[key].append(value)
    
    final_train_dic = dict(final_train_dic)

    
    #valid_loss_dic = {k: v.item() for k, v in valid_loss_dic.items()}
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
        print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
        total_test_loss += test_loss
        total_test_time += time.time() - start_time_test
    print('epoch %d total 3-px error in TEST = %.3f' %(epoch, total_test_loss/len(TestImgLoader)*100))
    print ("Total Test Time: ", str(total_test_time))
    
    test_loss_dic[epoch] = total_test_loss/len(TestImgLoader)
    test_time_dic[epoch] = total_test_time
    #test_loss_dic = {k: v.item() for k, v in test_loss_dic.items()}
    
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


    # if total_test_loss/len(TestImgLoader)*100 > max_acc:
    #     max_acc = total_test_loss/len(TestImgLoader)*100
    #     max_epo = epoch
    # print('MAX epoch %d total test error = %.3f' %(max_epo, max_acc))

	#    #SAVE
	#    savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
	#    torch.save({
	# 	    'epoch': epoch,
	# 	    'state_dict': model.state_dict(),
	# 	    'train_loss': total_train_loss/len(TrainImgLoader),
	# 	    'test_loss': total_test_loss/len(TestImgLoader)*100,
	# 	}, savefilename)
	
        
	# print(max_epo)
	# print(max_acc)


if __name__ == '__main__':
   main()
