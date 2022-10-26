# Hyper-GAN
This repository is the official Tensorflow implementation of the paper "A Unified Hyper-GAN Model for Unpaired Multi-contrast MR Image Translation" in MICCAI 2021.


## Overview

The outline of this readme file is:

    Overview
    Requirements
    Dataset
    Citation
    Reference
    
The folder structure of our implementation is:

    ours_model_fs\       : code of Hyper-GAN using filter scaling strategy
    ours_model_cin\      : code of Hyper-GAN using conditional instance normalization strategy
    data\                : root data folder (to be downloaded and preprocessed)





## Requirements





## Dataset
We use the Information eXtraction from Images (IXI) and MICCAI 2019 Multimodal Brain Tumor Segmentation (BraTS 2019) datasets in our experiments.

### Data Preprocessing

#### IXI Dataset


#### BraTS 2019 Dataset


### Data Folder Structure
The structure of our data folder is:

    data\    : root data folder  
        |-- IXI-Dataset\        : processed IXI data folder
        |       |-- SlicedData\      : processed 2D data in .npy format
        |       |       |-- Train\       : training data set
        |       |       |       |-- TrainA\       : t1w   images
        |       |       |       |-- TrainB\       : t2w   images
        |       |       |       |-- TrainC\       : pdw   images
        |       |-- VolumeData\      : processed 3D data in .nii.gz format
        |       |       |-- Valid\         : validation data set
        |       |       |       |-- ValidA\       : t1w   images
        |       |       |       |-- ValidB\       : t2w   images
        |       |       |       |-- ValidC\       : pdw   images
        |       |       |-- Test\          : test data set
        |       |       |       |-- TestA\        : t1w   images
        |       |       |       |-- TestB\        : t2w   images
        |       |       |       |-- TestC\        : pdw   images
        |-- BraTS-Dataset\      : processed BraTS 2019 data folder
        |       |-- SlicedData\      : processed 2D data in .npy format
        |       |       |-- Train\       : training data set
        |       |       |       |-- TrainA\       : t1w   images
        |       |-- VolumeData\      : processed 3D data in .nii.gz format
        |       |       |-- Valid\         : validation data set
        |       |       |       |-- ValidA\       : t1w   images
        |       |       |-- Test\          : test data set
        |       |       |       |-- TestA\        : t1w   images





## Citation
If you use this code for your research, please cite our paper:
> @inproceedings{yang2021unified,
> <br> title={A Unified Hyper-GAN Model for Unpaired Multi-contrast MR Image Translation},
> <br> author={Yang, Heran and Sun, Jian and Yang, Liwei and Xu, Zongben},
> <br> booktitle={MICCAI},
> <br> pages={127--137},
> <br> year={2021}
> <br> }


## Reference
