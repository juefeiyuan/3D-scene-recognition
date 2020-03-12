# Source Code of 3D Scene Recognition Methods DRF and sMVCNN
This is the source code repository for the methods proposed (DRF) or evaluated (sMVCNN) in the following paper:

"**Semantic Tree-Based 3D Scene Model Recognition**". MIPR 2020, PDF: http://orca.st.usm.edu/~bli/3DSceneRecognition/Semantic%20Tree-Based%203D%20Scene%20Model%20Recognition.pdf


## Data
**"27_additional_classes" folder**: the list of 27 additional manually annotated classes' names.\
**"probability_distributions" folder**: all the 30 scene categories' object occurrence probability distributions.

## Code
**DRF**: code for our deep random field (DRF) method. DRF shares the recognition framework with MVCNN but incorporates semantic information into the recognition process by integrating an additional semantic tree-based loss into the loss function definition.

**sMVCNN**: code for a baseline method scene based MVCNN (sMVCNN) which is modified from MVCNN (http://vis-www.cs.umass.edu/mvcnn/): changed the input
from 3D models to 3D scene models accordingly to run on the benchmark in our experiments. 

## Dataset
Please download the 3D scene benchmark **Scene_SBR_IBR directly** via its dataset webpage. 

**Scene_SBR_IBR**: http://orca.st.usm.edu/~bli/Scene_SBR_IBR/.


## Trained Models
The trained models include: 
1. The Trained sMVCNN model started with pre-trained Places model.
2. The Trained DRF model started with pre-trained Places model
3. The Trained sMVCNN model started with randeomly initialized Places model
4. The Trained DRF model started with randeomly initialized Places model

You can obtain the trained model files by directly downloading via this respository 

**Trained models**: http://orca.st.usm.edu/~bli/3DSceneRecognition/data.html

