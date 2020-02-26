# PyTorch code for sMVCNN  
Code is tested on Python 3.6 and PyTorch 1.0.1
Code is a baseline code based on MVCNN(http://vis-www.cs.umass.edu/mvcnn/). The difference is that the input is 3D scene data, not 3D object data.

First, download benchmark and put the sampled images under ```modelnet40_images_new_12x```. 
Then, run the 'train_mvcnn.py' file.

'models' file contains the model initialization, which is used for training.
