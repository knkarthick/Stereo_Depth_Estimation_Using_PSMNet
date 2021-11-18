import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
#import preprocess 
from . import preprocess 
import cv2  

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
           width, height = left_img.size
           #print (width, height)
           left_img = cv2.pyrMeanShiftFiltering(src=np.asarray(left_img), sp=15, sr=20) #added         
           number_of_grey_black_pix_pct = (np.sum(left_img < 50)) / (width * height)
           if 0.50 < number_of_grey_black_pix_pct < 0.75:
               left_img = adjust_gamma(left_img, 1)
           elif number_of_grey_black_pix_pct > 0.5:
               left_img = adjust_gamma(left_img, 1)
           else:
               pass
           left_img=Image.fromarray(left_img)
           enhancer = ImageEnhance.Sharpness(left_img)
           left_img = enhancer.enhance(1)
                     
           right_img = cv2.pyrMeanShiftFiltering(src=np.asarray(right_img), sp=15, sr=20) #added
           number_of_grey_black_pix_pct = (np.sum(right_img < 50)) / (width * height)
           if 0.50 < number_of_grey_black_pix_pct < 0.75:
                right_img = adjust_gamma(right_img, 1)
           elif number_of_grey_black_pix_pct > 0.5:
                right_img = adjust_gamma(right_img, 1)
           else:
                pass
        
           right_img=Image.fromarray(right_img)
           enhancer = ImageEnhance.Sharpness(right_img)
           right_img = enhancer.enhance(1)
           width, height = right_img.size
           #print (width, height)
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, dataL
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-1232, h-368, w, h))
           right_img = right_img.crop((w-1232, h-368, w, h))
           w1, h1 = left_img.size

           dataL = dataL.crop((w-1232, h-368, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

           processed = preprocess.get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
