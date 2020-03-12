# PyTorch code for deep random field (DRF) 
Code is tested on Python 3.6 and PyTorch 1.0.1

**Usage**: 
First, download the Scene_SBR_IBR dataset via http://orca.st.usm.edu/~bli/Scene_SBR_IBR/;

Then, sample the view images for each 3D scene model, and then put the sampled images into the 'modelnet40_images_new_12x' folder. 

Finally, run the 'train_mvcnn.py' file for training . The 'models' folder contains the model initialization, which is used for training.
