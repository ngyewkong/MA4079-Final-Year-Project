# MA4079-Final-Year-Project
Breast Tumour Segmentation Using Convolutional Neural Network on 3D Computed Tomography (CT) Images

Python Code for U-Net is inspired by Dennis Madsen's implementation of 3D Volumetric Segmentation using 2D U-Net. Link to his github repo: https://github.com/madsendennis/notebooks/tree/master/volume_segmentation_with_unet

- Achieved DICE score of 79.2% on training set, 78.4% on test set.

Python Code for U-Net with pretrained MobileNetV2 encoder is inspired by Nikhil Tomar's implementation. Link to his github repo: https://github.com/nikhilroxtomar/Unet-with-Pretrained-Encoder

- Achieved DICE score of 91.7% on training set, 86.9% on test set.

# Data Preparation
Dataset given are patients' 3D CT volumes under a collaboration with National Cancer Centre Singapore (NCCS). Due to Patients' Personal Data Privacy Agreement (PDPA) Law in Singapore, Dataset is not available to public. Dataset is hosted on a private Nanyang Technological University (NTU) server, with GPU support.

Dataset consists of 247 patients' volumes in nrrd data format. 

In order to create 2D slices from 3D volumes, a series of pre-processing steps are needed. 
  1. Using pynrrd library to read nrrd volumes 
  2. Run python script to implement slicing for the whole dataset

# Model Setup
In order to train the model, the 2D slices need to be put into the respective directories for tensorflow & keras to work properly.
Hence, some linux commands are needed as the whole dataset and training is hosted remotely.

# Model Training & Evaluation

# Results Discussion 
Segmentation Results (pending approval for open public access)
