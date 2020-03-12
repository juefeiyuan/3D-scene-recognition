# PyTorch code for DRF 
Code is tested on Python 3.6 and PyTorch 1.0.1

**sMVCNN**: code for a baseline method scene based MVCNN (sMVCNN) which is modified from MVCNN (http://vis-www.cs.umass.edu/mvcnn/): changed the input
from 3D models to 3D scene models accordingly to run on the benchmark in our experiments. Code has been tested on Python 3.6 and PyTorch 1.0.1.

**Usage**: 
First, download the Scene_SBR_IBR dataset via http://orca.st.usm.edu/~bli/Scene_SBR_IBR/;

Then, sample the view images for each 3D scene model, and then put the sampled images into the 'modelnet40_images_new_12x' folder. 

Finally, run the 'train_mvcnn.py' file for training. The 'models' folder contains the model initialization, which is used for training.
