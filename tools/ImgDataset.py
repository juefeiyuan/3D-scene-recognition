import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random

class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True):
        self.classnames=['airport_terminal','apartment_building_outdoor','arch','auditorium','barn','beach',
                         'bedroom','castle','classroom',
                         'conference_room','dam','desert','football_stadium','great_pyramid','hotel_room','kitchen','library',
                         'mountain','office','phone_booth','reception','restaurant','river','school_house',
                         'shower','skyscraper','supermarket','waiting_room','water_tower','windmill']

        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.root= 'D:/JuefeiYuan/TensorFlowFiles/SHREC2018_Plus_VGG/PytorchVersion/Multi_View/mvcnn_pytorch-master_SHREC/'########
        parent_dir = self.root + parent_dir #######
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            ## Select subset for different number of views
            stride = int(12/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new


        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.Resize([224, 224]),############
                #transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.Resize([224, 224]),############
                #transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        #class_name = path.split('/')[-3]
        class_name = path.split('/')[-2]###############
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])



class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12):
        self.classnames=['airport_terminal','apartment_building_outdoor','arch','auditorium','barn','beach',
                         'bedroom','castle','classroom',
                         'conference_room','dam','desert','football_stadium','great_pyramid','hotel_room','kitchen','library',
                         'mountain','office','phone_booth','reception','restaurant','river','school_house',
                         'shower','skyscraper','supermarket','waiting_room','water_tower','windmill']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.root= 'D:/JuefeiYuan/TensorFlowFiles/SHREC2018_Plus_VGG/PytorchVersion/Multi_View/mvcnn_pytorch-master_SHREC/'########
        parent_dir = self.root + parent_dir #######
        self.filepaths = []
       
        #sb1=glob.glob('D:/JuefeiYuan/TensorFlowFiles/SHREC2018_Plus_VGG/PytorchVersion/Multi_View/mvcnn_pytorch-master/tools/modelnet40_images_new_12x/airplane/train/*.png')#加上r让字符串不转义
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),############
            #transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]
        #class_name = path.split('/')[-3]
        class_name = path.split('/')[-2]############
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return (class_id, im, path)
