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

* [IXI dataset](https://brain-development.org/ixi-dataset/): This dataset includes nearly 600 MR images from normal healthy subjects, which were collected at three different hospitals in London. In this experiment, we utilize all 319 subjects from Guy’s Hospital, and randomly split them into 150, 5 and 164 subjects for training, validation and test. Each subject contains three contrasts (T1w, T2w and PDw), and only one of three contrasts per subject is used for training to generate unpaired data.

* [BraTS 2019 dataset](https://www.med.upenn.edu/cbica/brats-2019/): This dataset includes 335 training subjects and 125 validation subjects, which were acquired with different clinical protocols and various scanners from multiple (n=19) institutions. In our experiments, we use all 150 subjects from CBICA institution, and randomly split them into 100, 5 and 45 subjects for training, validation and test. Each subject contains four contrasts (T1w, T1Gd, T2w, FLAIR), and only one of four contrasts per subject is used for training.

### Data Preprocessing

#### IXI Dataset


#### BraTS 2019 Dataset
The data has been pre-processed by organizers, i.e., co-registered to the same anatomical template, interpolated to the same resolution and skull-stripped.
Additionally, we conduct several extra pre-processing steps:
* N4 correction
* White matter peak normalization of each modality to 1000
* Cutting out the black background area outside the brain
After preprocessing, the maximal intensities of T1w, T1ce, T2w and Flair modalities are 3000, 5000, 6000 and 7000 (arbitrary units) respectively.
To reduce the time of loading data, we save the processed training data in .npy format, and save the processed validation and test data in .nii.gz format.



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
