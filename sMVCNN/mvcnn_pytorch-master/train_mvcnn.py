import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse


from Trainer import ModelNetTrainer

from ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN


name='mvcnn'
num_models=1000
weight_decay=0.001
num_views=13
train_path='modelnet40_images_new_12x/*/train'
val_path='modelnet40_images_new_12x/*/test'
cnn_name='vgg16'
lr=5e-5
batchSize=8

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    # Load in converted weights into the untrained model 
    #places365_weights = torch.load('vgg16_places365.pt')#('vgg_places365.pth')
    #vgg.load_state_dict(places365_weights)
    #vgg.load_state_dict({l : torch.from_numpy(numpy.array(v)).view_as(p) for k, v in places365_weights.items() for l, p in vgg.named_parameters() if k in l})########

    #pretraining = not args.no_pretraining
    pretraining=False

    log_dir = name

    create_folder(name)


    # STAGE 1
    log_dir = name+'_stage_1'
    create_folder(log_dir)
    #cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)
    cnet = SVCNN(name, nclasses=30, pretraining=pretraining, cnn_name=cnn_name)

   
    optimizer = optim.Adam(cnet.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    n_models_train = num_models*num_views

    
    train_dataset = SingleImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    val_dataset = SingleImgDataset(val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    best_acc_coarse=[]
    for i in range(1, 9):
        trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views, LAMBDA_val=i/10)
        
        trainer.train(50)

        # STAGE 2
        log_dir = name+'_stage_2'
        create_folder(log_dir)
 
        cnet_2 = MVCNN(name, cnet, nclasses=30, cnn_name=cnn_name, num_views=num_views)
        del cnet

        
        optimizer = optim.Adam(cnet_2.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        
        
        train_dataset = MultiviewImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

        
        val_dataset = MultiviewImgDataset(val_path, scale_aug=False, rot_aug=False, num_views=num_views)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=0)
        print('num_train_files: '+str(len(train_dataset.filepaths)))
        print('num_val_files: '+str(len(val_dataset.filepaths)))
        
        trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=num_views)
   
        tmp=trainer.train(50)
        best_acc_coarse.append(tmp)

    best_i=np.argmax(best_acc_coarse)
    best_acc_fine=[]
    for i in range(best_i*10-9, best_i*10+10):
        trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views, LAMBDA_val=i/100)
       
        trainer.train(50)

        # STAGE 2
       
        log_dir = name+'_stage_2'
        create_folder(log_dir)
    
        cnet_2 = MVCNN(name, cnet, nclasses=30, cnn_name=cnn_name, num_views=num_views)
        del cnet

        
        optimizer = optim.Adam(cnet_2.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        
        
        train_dataset = MultiviewImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

        
        val_dataset = MultiviewImgDataset(val_path, scale_aug=False, rot_aug=False, num_views=num_views)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=0)
        print('num_train_files: '+str(len(train_dataset.filepaths)))
        print('num_val_files: '+str(len(val_dataset.filepaths)))
        
        trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=num_views)
        
        tmp=trainer.train(50)
        best_acc_fine.append(tmp)


