# Hyper-GAN
This repository is the official Tensorflow implementation of the paper "A Unified Hyper-GAN Model for Unpaired Multi-contrast MR Image Translation" in MICCAI 2021.


## Overview

<div align=center>
<img src="https://github.com/HeranYang/Hyper-GAN/blob/main/images/framework.png" width="700px"/>
</div>
    
    
We propose a unified Hyper-GAN model for effectively and efficiently translating between different contrast pairs. Hyper-GAN consists of a pair of hyper-encoder and hyper-decoder to first map from the source contrast to a common feature space, and then further map to the target contrast image. To facilitate the translation between different contrast pairs, contrast-modulators are designed to tune the hyper-encoder and hyper-decoder adaptive to different contrasts. We also design a common
space loss to enforce that multi-contrast images of a subject share a common feature space, implicitly modeling the shared underlying anatomical structures. Experiments on two datasets of IXI and BraTS 2019 show that our Hyper-GAN achieves state-of-the-art results in both accuracy and efficiency, e.g., improving more than 1.47 and 1.09 dB in PSNR on two datasets with less than half the amount of parameters.





The outline of this readme file is:

    Overview
    Requirements
    Dataset
    Usage
    Citation
    
The folder structure of our implementation is:

    ours_model_fs\       : code of Hyper-GAN using filter scaling strategy
    ours_model_cin\      : code of Hyper-GAN using conditional instance normalization strategy
    data\                : root data folder (to be downloaded and preprocessed)





## Requirements
All experiments utilize the TensorFlow library. We recommend the following package versions:
* python == 3.6
* tensorflow-gpu == 1.10.0
* numpy == 1.19.2
* imageio == 2.9.0
* nibabel == 3.2.1




## Dataset
We use the Information eXtraction from Images (IXI) dataset and MICCAI 2019 Multimodal Brain Tumor Segmentation (BraTS 2019) dataset in our experiments.

* [IXI dataset](https://brain-development.org/ixi-dataset/): This dataset includes nearly 600 MR images from normal healthy subjects, which were collected at three different hospitals in London. In this experiment, we utilize all 319 subjects from Guy’s Hospital, and randomly split them into 150, 5 and 164 subjects for training, validation and test. Each subject contains three contrasts (T1w, T2w and PDw), and only one of three contrasts per subject is used for training to generate unpaired data.

* [BraTS 2019 dataset](https://www.med.upenn.edu/cbica/brats-2019/): This dataset includes 335 training subjects and 125 validation subjects, which were acquired with different clinical protocols and various scanners from multiple (n=19) institutions. In our experiments, we use all 150 subjects from CBICA institution, and randomly split them into 100, 5 and 45 subjects for training, validation and test. Each subject contains four contrasts (T1w, T1Gd, T2w, FLAIR), and only one of four contrasts per subject is used for training.

### Data Preprocessing

To reduce the time of loading data, we save the processed training data in .npy format, and save the processed validation and test data in .nii.gz format.

#### IXI Dataset
For this dataset, We conduct multiple careful data pre-processing steps for the T1w, T2w and PDw images:
* N4 correction
* Isotropic to 1.0mm^3 (only for T2w)
* Affine registration (for T1w, PDw to the T2w space)
* N4 correction
* White matter peak normalization of each modality to 1000
* Pad the image volumes to 240x256x192
* Pick boundary slice ID

After preprocessing, the maximal intensities of T1w, T2w and PDw modalities are 3000, 6000 and 3000 (arbitrary units) respectively.
For the training subset, the intensities are further linearly scaled to [0, 1], and then the processed 2d training images are saved in .npy format.
(The linear scaling for validation and test subsets is included in our codes within utils.py.)



#### BraTS 2019 Dataset
The data has been pre-processed by organizers, i.e., co-registered to the same anatomical template, interpolated to the same resolution and skull-stripped.
Additionally, we conduct several extra pre-processing steps:
* N4 correction
* White matter peak normalization of each modality to 1000
* Pick boundary slice ID
* Pad the 2D sagittal images to 240x160

After preprocessing, the maximal intensities of T1w, T1ce, T2w and Flair modalities are 3000, 5000, 6000 and 7000 (arbitrary units) respectively.
For the training subset, the intensities are further linearly scaled to [0, 1], and then the processed 2d training images are saved in .npy format.
(The linear scaling for validation and test subsets is included in our codes within utils.py.)





### Data Folder Structure
The structure of our data folder is:

    data\    : root data folder  
        |-- IXI-Dataset\     : processed IXI data folder
        |       |-- IXI-dataInfo.txt       : save the boundary slice id
        |       |-- SlicedData\            : processed 2D data in .npy format
        |       |       |-- Train\            : training data set
        |       |       |       |-- TrainA\       : t1w   images
        |       |       |       |       |-- DomA{:0>3d}-slice{:0>3d}.npy       : image name format
        |       |       |       |-- TrainB\       : t2w   images
        |       |       |       |       |-- DomB{:0>3d}-slice{:0>3d}.npy       : image name format
        |       |       |       |-- TrainC\       : pdw   images
        |       |       |       |       |-- DomC{:0>3d}-slice{:0>3d}.npy       : image name format
        |       |-- VolumeData\            : processed 3D data in .nii.gz format
        |       |       |-- Valid\            : validation data set
        |       |       |       |-- ValidA\       : t1w   images
        |       |       |       |       |-- DomA{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- ValidB\       : t2w   images
        |       |       |       |       |-- DomB{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- ValidC\       : pdw   images
        |       |       |       |       |-- DomC{:0>3d}.nii.gz                 : image name format
        |       |       |-- Test\             : test data set
        |       |       |       |-- TestA\        : t1w   images
        |       |       |       |       |-- DomA{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- TestB\        : t2w   images
        |       |       |       |       |-- DomB{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- TestC\        : pdw   images
        |       |       |       |       |-- DomC{:0>3d}.nii.gz                 : image name format
        |-- BraTS-Dataset\    : processed BraTS 2019 data folder
        |       |-- BraTS-dataInfo.txt    : save the boundary slice id
        |       |-- SlicedData\           : processed 2D data in .npy format
        |       |       |-- Train\            : training data set
        |       |       |       |-- TrainA\       : t1w   images
        |       |       |       |       |-- DomA{:0>3d}-slice{:0>3d}.npy       : image name format
        |       |       |       |-- TrainB\       : t1ce  images
        |       |       |       |       |-- DomB{:0>3d}-slice{:0>3d}.npy       : image name format
        |       |       |       |-- TrainC\       : t2w   images
        |       |       |       |       |-- DomC{:0>3d}-slice{:0>3d}.npy       : image name format
        |       |       |       |-- TrainD\       : flair images
        |       |       |       |       |-- DomD{:0>3d}-slice{:0>3d}.npy       : image name format
        |       |-- VolumeData\           : processed 3D data in .nii.gz format
        |       |       |-- Valid\            : validation data set
        |       |       |       |-- ValidA\       : t1w   images
        |       |       |       |       |-- DomA{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- ValidB\       : t1ce  images
        |       |       |       |       |-- DomB{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- ValidC\       : t2w   images
        |       |       |       |       |-- DomC{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- ValidD\       : flair images
        |       |       |       |       |-- DomD{:0>3d}.nii.gz                 : image name format
        |       |       |-- Test\             : test data set
        |       |       |       |-- TestA\        : t1w   images
        |       |       |       |       |-- DomA{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- TestB\        : t1ce  images
        |       |       |       |       |-- DomB{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- TestC\        : t2w   images
        |       |       |       |       |-- DomC{:0>3d}.nii.gz                 : image name format
        |       |       |       |-- TestD\        : flair images
        |       |       |       |       |-- DomD{:0>3d}.nii.gz                 : image name format



## Usage

We provide our Hyper-GAN codes on BraTS 2019 dataset, and you can extend to your own dataset by modifying the configs in main.py, the dataset path in model.py, and the intensity ranges in utils.py, etc.

The structure of our code folder is:

    ours_model_fs\       : code of Hyper-GAN using filter scaling strategy
           |-- main.py         : main function
           |-- model.py        : code of building model, and train/valid/test
           |-- module.py       : code of defining networks
           |-- ops.py          : code of defining basic components
           |-- utils.py        : code of loading train and test data
    ours_model_cin\      : code of Hyper-GAN using conditional instance normalization strategy
           |-- main.py         : main function
           |-- model.py        : code of building model, and train/valid/test
           |-- module.py       : code of defining networks
           |-- ops.py          : code of defining basic components
           |-- utils.py        : code of loading train and test data



### Training

Our code can be trained using the following commond:

    python main.py --phase=train

If you want to continue train the model, you could uncomment the continue_training codes in train function within model.py, and then run the commond above.


### Validation

Before starting the validation process, you may need to modify the information about valid set and epoch in valid function within model.py.
Then, the validation process can be conducted using the following commond:

    python main.py --phase=valid
    
After generating the validation results, you could select the optimal epoch_id based on the performance on validation set.


### Test

Before starting the test process, you need to set the epoch as the selected optimal epoch_id in test function within model.py.
Then, you can generate the test results using the following commond:

    python main.py --phase=test


#### About Trained Model

We have uploaded our trained Hyper-GAN models on BraTS 2019 dataset, and one can directly use them for multi-contrast MR image synthesis.
Due to the size restriction of upload files in github, the models are uploaded to 
the [Google Drive](https://drive.google.com/drive/folders/1phlXLZBMaWr6UcWACppv1fvTkOTqfFXK?usp=share_link) 
and [Baidu Netdisk](https://pan.baidu.com/s/1PeHPjrhDsPUcvXncR1JftA?pwd=ffr4).




## Citation
If you find this code useful for your research, please cite our paper:
> @inproceedings{yang2021unified,
> <br> title={A Unified Hyper-GAN Model for Unpaired Multi-contrast MR Image Translation},
> <br> author={Yang, Heran and Sun, Jian and Yang, Liwei and Xu, Zongben},
> <br> booktitle={MICCAI},
> <br> pages={127--137},
> <br> year={2021}}


