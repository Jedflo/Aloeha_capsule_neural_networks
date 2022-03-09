# Aloeha_capsnet

## Introduction 
Aloeha is an undergradaute thesis by\
Joash Edson Flores (Me)\
[Raymond Gatdula](https://github.com/Exqst)\
[Mathew Mojado](https://github.com/MachuMachu)\
[Rudolph Vincent Sta. Maria](https://github.com/raze47)

Aloeha Capsnet is a Tensorflow 2.2 implementation of [Capusle Neural Network by Frosst, Hinton, And sabour](https://link-url-here.org). The code used is based on 
[Xifeng Guo's Capsnet for tensorflow 2.2](https://github.com/XifengGuo/CapsNet-Keras/tree/tf2.2), which was adapted for and trained on the [Aloe Vera images dataset](https://www.kaggle.com/rubab123/aloe-vera-images)

The output of the thesis is an [Android application](https://github.com/Jedflo/Aloeha_UI_Manager) that uses capsule neural networks to identify if an aloe vera leaf is healthy, rotting or rusting. 
The goal is to measure the performance of capsnet in identifying aloe vera leaf afflications and compare it to the performance of [Aloeha-Alexnet](https://github.com/raze47/AlexNet-Aloeha) and [Aloeha-VGG19](https://github.com/raze47/VGG19-Aloeha)

## Aloeha Capsnet Model and Training
### The model architecture
![image](https://user-images.githubusercontent.com/82581503/156983510-3109d1e1-0ec7-40e0-878f-c4c39ab1bebd.png)
 Layers:
 1. Input Layer
    * Input images must be *32 x 32 pixels* and in *RGB color mode* 
 2. Convolutional Layer: Normal Convolutional Layer
    * 32 Filters 
    * 9 x 9 kernel size
    * stride of 1
    * has padding
    * uses **leaky ReLu** activation functions with an *alpha of 0.3*
 3. Primary Capsule Layer: Another Conv layer + squash function + reshape to vectors
    * 4 dimension capsules
    * 8 channels 
    * 9 x 9 kernel size
    * stride of 2
    * has padding
 4. Aloe Capsules Layer: Layer where dynamic routing occurs
    * has 3 capsules
    * each capsule has 8 dimensions
    * 3 rounds of dynamic routing
 5. Output Layer: Produces classifications/predictions
 6. Decoder Network: Produces reconstructed images from the input. This part is optional, but based on our experiments, a decoder network improves capsnet accuracy.
    * Consists of 3 dense layers. 2 of which are Leaky ReLu layers and 1 sigmoid layer. 

### Training  
Training hyper parameters used:
   * trained for 200 epochs
   * with batch size of 20
   * learning rate of 0.001
   * learning rate will be multiplied by 0.9 if the validation accuracy did not improve by 0.0001% within 3 epochs.


## Results
We trained the model on a computer with a GTX 1050 GPU. Each epoch took ~40 seconds.

### Performance during training:

![capsnet graph](https://user-images.githubusercontent.com/82581503/156989638-ab8b0a25-83cd-494c-a2e0-836b078780b7.png)

### Performance during testing:


0 = Healthy Class of Aloe Vera Leaves \
1 = Rotting Class of Aloe Vera Leaves \
2 = Rusting Class of Aloe Vera Leaves 

**Precison** is the ratio of correctly predicted classifications over the total number of predicted classifications.

**Recall** is the ratio of correctly predicted classifications over the total number of actual classifications.

**F1 score** is the harmonic mean of precision and recall.

Note that Precision, Recall, and F1 score is computed per class, unlike accuracy, which considers all classes. 

![performance metrics](https://user-images.githubusercontent.com/82581503/156989725-55d0803e-5a21-4393-bbee-e4ff4e1ae03e.png)

### Reconstructions
from top to bottom: the first 5 rows are images from the dataset while the following rows are capsnet reconstructions. 

![real_and_recon](https://user-images.githubusercontent.com/82581503/157379791-69a30fa3-839d-41c1-a43a-e9c439f81825.png)

Aloeha capsnet, unlike MNIST Capsnet, was not able to reconstruct the input images close to the original. This can be due to the small image sizes we used. The decision to use 32 x 32 Aloe vera images was caused by the high computational power it would require to train the model on larger images. 

### Confusion Matrix

*Classified as* means that Aloeha Capsnet has classified a given input image as healty, rot, or rust \
*Expected* is the true classification of a given input image.

![image](https://user-images.githubusercontent.com/82581503/157383577-9c8d76c9-8336-4229-b9a7-cf8e5adc6b9d.png)

Note that the numbers above represent *number of images*. i.e., 1 image was classified as containing a healthy leaf, but the leaf actually had rot.  




