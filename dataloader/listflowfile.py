import torch.utils.data as data

from PIL import Image
import os
import os.path
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    print (classes)
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    print (image)
    disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]
    print (disp)
    

    # monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    # monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]


    # monkaa_dir  = os.listdir(monkaa_path)

    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []


    # for dd in monkaa_dir:
    #   for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
    #     if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
    #       all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
    #     all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')

    #   for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
    #     if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
    #       all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

    # flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
    # flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
    # flying_dir = flying_path+'/TRAIN/'
    # subdir = ['A','B','C']

    # for ss in subdir:
    #   flying = os.listdir(flying_dir+ss)

    #   for ff in flying:
    #     imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #     for im in imm_l:
    #       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #         all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
    #       all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

    #       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #         all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    # flying_dir = flying_path+'/TEST/'

    # subdir = ['A','B','C']

    # for ss in subdir:
    #   flying = os.listdir(flying_dir+ss)

    #   for ff in flying:
    #     imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #     for im in imm_l:
    #       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #         test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
    #       test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

    #       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #         test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)



    driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    driving_disp = filepath + [x for x in disp if 'driving' in x][0]
    print (driving_dir)
    print (driving_disp)

    
    subdir1 = ['35mm_focallength','15mm_focallength']
    subdir2 = ['scene_backwards','scene_forwards']
    subdir3 = ['fast','slow']

    for i in subdir1:
      for j in subdir2:
        for k in subdir3:
            imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')    
            for im in imm_l:
              if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)

              all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

              if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)

    print ("Before ")
    print (len(all_left_img))
    split = (int(len(all_left_img)*0.15))
    print (split)
    random_index = random.sample(range(0, len(all_right_img)), split)
    #print (random_index)
    for i in range(len(random_index)):
      test_left_img.append(all_left_img[random_index[i]])
      test_right_img.append(all_right_img[random_index[i]])
      test_left_disp.append(all_left_disp[random_index[i]])

    for i in sorted(random_index, reverse=True):
      del all_left_img[i]
      del all_right_img[i]
      del all_left_disp[i]
    
    print ("After")
    print (len(all_left_img))
    print(len(test_left_disp))
    print (test_left_img[:2])
    print (test_right_img[:2])
    print (test_left_disp[:2])


    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


