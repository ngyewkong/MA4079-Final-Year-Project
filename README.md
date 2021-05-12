# MA4079-Final-Year-Project
Breast Tumour Segmentation Using Convolutional Neural Network on 3D Computed Tomography (CT) Images

Python Code for U-Net is inspired by Dennis Madsen's implementation of 3D Volumetric Segmentation using 2D U-Net. Link to his github repo: https://github.com/madsendennis/notebooks/tree/master/volume_segmentation_with_unet

![image](https://user-images.githubusercontent.com/55376202/117907443-68d81000-b309-11eb-99fe-a091cb7f7ffc.png)


- Best Model achieved Dice score of 84.53% on training set, 85.65% on test set.

Python Code for U-Net with pretrained MobileNetV2 encoder is inspired by Nikhil Tomar's implementation. Link to his github repo: https://github.com/nikhilroxtomar/Unet-with-Pretrained-Encoder

- Best Model achieved Dice score of 88.19% on training set, 87.9% on test set.

Python Code for Attention-Guided U-Net is derived from 

@inproceedings{noori2019attention,
  title={Attention-Guided Version of 2D UNet for Automatic Brain Tumor Segmentation},
  author={Noori, Mehrdad and Bahri, Ali and Mohammadi, Karim},
  booktitle={2019 9th International Conference on Computer and Knowledge Engineering (ICCKE)},
  pages={269--275},
  year={2019},
  organization={IEEE}
}
 Link to his github repo: https://github.com/Mehrdad-Noori/Brain-Tumor-Segmentation
 
 - Best Model achieved Dice Score of 86.02% on training set, 83.71% on test set.

# Data Preparation
Dataset given are patients' 3D CT volumes under a collaboration with National Cancer Centre Singapore (NCCS). Due to Patients' Personal Data Privacy Agreement (PDPA) Law in Singapore, Dataset is not available to public. Dataset is hosted on a private Nanyang Technological University (NTU) server, with GPU support.

Dataset consists of 347 patients' volumes in nrrd data format.

Image Normalisation is first carried out as the image dataset is in CT modality, which uses Hounsfield Units. Image dataset is converted to a 0 to 1 scale.

In order to create 2D slices from 3D volumes, a series of pre-processing steps are needed. 
  1. Using pynrrd library to read nrrd volumes 
  2. Run python script to implement slicing for the whole dataset
 
 Data Augmentation is also applied using Image Data Generator in Keras.

# Model Setup
In order to train the model, the 2D slices need to be put into the respective directories for tensorflow & keras to work properly.
Hence, some linux commands are needed as the whole dataset and training is hosted remotely.

This study adopts a 70-30 train-test split, image input shape of 256 by 256. 

# Hyperparameters 
Batch Size: 2
Learning Rate: 0.0003 
Optimiser: Adam with Nesterov Momentum 
Epochs: 100

# Model Training & Evaluation
3 Loss Functions tested in this study:
  1. Binary Cross Entropy Loss (BCE)
  2. Dice Loss
  3. Combination Loss (BCE + Dice)

# Results Discussion 
Due to PDPA agreement: Results containing patients' data and implementation code are not shown
