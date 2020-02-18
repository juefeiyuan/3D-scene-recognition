import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

#from tools.Trainer import ModelNetTrainer
from Trainer import ModelNetTrainer
#from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

# parser = argparse.ArgumentParser()
# parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
# parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
# parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
# parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
# parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
# parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
# parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
# parser.add_argument("-num_views", type=int, help="number of views", default=12)
# parser.add_argument("-train_path", type=str, default="modelnet40_images_new_12x/*/train")
# parser.add_argument("-val_path", type=str, default="modelnet40_images_new_12x/*/test")
# parser.set_defaults(train=False)

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
    #args = parser.parse_args()

    # Load in converted weights into the untrained model 
    #places365_weights = torch.load('vgg16_places365.pt')#('vgg_places365.pth')
    #vgg.load_state_dict(places365_weights)
    #vgg.load_state_dict({l : torch.from_numpy(numpy.array(v)).view_as(p) for k, v in places365_weights.items() for l, p in vgg.named_parameters() if k in l})########

    #pretraining = not args.no_pretraining
    pretraining=False
    #log_dir = args.name
    log_dir = name
    #create_folder(args.name)
    create_folder(name)
    #config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    #json.dump(vars(args), config_f)
    #config_f.close()

    # STAGE 1
    #log_dir = args.name+'_stage_1'
    log_dir = name+'_stage_1'
    create_folder(log_dir)
    #cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)
    cnet = SVCNN(name, nclasses=30, pretraining=pretraining, cnn_name=cnn_name)

    #optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(cnet.parameters(), lr=lr, weight_decay=weight_decay)
    
    #n_models_train = args.num_models*args.num_views
    n_models_train = num_models*num_views

    #train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_dataset = SingleImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    val_dataset = SingleImgDataset(val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

    best_acc_coarse=[]
    for i in range(1, 9):
        trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views, LAMBDA_val=i/10)
        #trainer.train(30)
        trainer.train(1)

        # STAGE 2
        #log_dir = args.name+'_stage_2'
        log_dir = name+'_stage_2'
        create_folder(log_dir)
        #cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
        cnet_2 = MVCNN(name, cnet, nclasses=30, cnn_name=cnn_name, num_views=num_views)
        del cnet

        #optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        optimizer = optim.Adam(cnet_2.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        
        #train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
        train_dataset = MultiviewImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

        #val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
        val_dataset = MultiviewImgDataset(val_path, scale_aug=False, rot_aug=False, num_views=num_views)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=0)
        print('num_train_files: '+str(len(train_dataset.filepaths)))
        print('num_val_files: '+str(len(val_dataset.filepaths)))
        #trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
        trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=num_views)
        #trainer.train(30)
        tmp=trainer.train(1)
        best_acc_coarse.append(tmp)

    best_i=np.argmax(best_acc_coarse)
    best_acc_fine=[]
    for i in range(best_i*10-9, best_i*10+10):
        trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views, LAMBDA_val=i/100)
        #trainer.train(30)
        trainer.train(50)

        # STAGE 2
        #log_dir = args.name+'_stage_2'
        log_dir = name+'_stage_2'
        create_folder(log_dir)
        #cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
        cnet_2 = MVCNN(name, cnet, nclasses=30, cnn_name=cnn_name, num_views=num_views)
        del cnet

        #optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        optimizer = optim.Adam(cnet_2.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        
        #train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
        train_dataset = MultiviewImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

        #val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
        val_dataset = MultiviewImgDataset(val_path, scale_aug=False, rot_aug=False, num_views=num_views)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=0)
        print('num_train_files: '+str(len(train_dataset.filepaths)))
        print('num_val_files: '+str(len(val_dataset.filepaths)))
        #trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
        trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=num_views)
        #trainer.train(30)
        tmp=trainer.train(50)
        best_acc_fine.append(tmp)


