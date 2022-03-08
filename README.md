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

![capsnet graph](https://user-images.githubusercontent.com/82581503/156989638-ab8b0a25-83cd-494c-a2e0-836b078780b7.png)
The model was trained for 200 Epochs. 

![performance metrics](https://user-images.githubusercontent.com/82581503/156989725-55d0803e-5a21-4393-bbee-e4ff4e1ae03e.png)

![conf matrix](https://user-images.githubusercontent.com/82581503/156989788-34036c61-3d30-40d1-a767-f862c0e3ed15.png)

