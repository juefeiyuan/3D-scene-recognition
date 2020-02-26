# 3D Scene Recognition Methods Source Code
This is the source code repository for the methods evaluated in the following manuscript: 

"**Semantic Tree-Based 3D Scene Model Recognition**". MIPR 2020, PDF: http://orca.st.usm.edu/~bli/3DSceneRecognition/data.html


## Data
27_additional_classes: It contains the list of 27 additional manually annotated classes' names.
probability_distributions: It contains all the 30 scene categories' object occurrence probability distributions.

## Code
sMVCNN: This file contains a baseline code based on MVCNN(http://vis-www.cs.umass.edu/mvcnn/), we changed the input data from 3D shape model to 3D scene model so as to get a baseline result.
DRF: This file incorporates our deep random field(DRF) method, DRF shares with MVCNN in terms of the recognition framework but incorporates the additional semantic-tree based loss into the loss function definition.

You can obtain the code files by directly downloading via this respository 

**Code**: http://orca.st.usm.edu/~bli/3DSceneRecognition/data.html

## Trained Models
The trained models include: 
1. The Trained sMVCNN model started with pre-trained Places model.
2. The Trained DRF model started with pre-trained Places model
3. The Trained sMVCNN model started with randeomly initialized Places model
4. The Trained DRF model started with randeomly initialized Places model

You can obtain the trained model files by directly downloading via this respository 

**Trained models**: http://orca.st.usm.edu/~bli/3DSceneRecognition/data.html

## Benchmark
You can obtain the 3D scene benchmark by by accessing the following dataset webpage.  

**Benchmark**: http://orca.st.usm.edu/~bli/Scene_SBR_IBR/