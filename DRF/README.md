# PyTorch code for deep random field (DRF) 
Code is tested on Python 3.6 and PyTorch 1.0.1

**Usage**: \
First, download the Scene_SBR_IBR dataset(same as sMVCNN);

Then, sample the view images and put them into the 'modelnet40_images_new_12x' folder(same as sMVCNN). 

Finally, run the 'train_mvcnn.py' file for training. Compared to sMVCNN, the 'Trainer.py' file incorporates the semantically relatedness relationships between the detected 
scene objects’ labels and the candidate scene category labels into the training and prediction.
