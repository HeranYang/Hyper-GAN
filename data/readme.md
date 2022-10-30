This folder is the root data folder containing processed dataset, which needs to be downloaded and processed by users. The structure of our data folder is:

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


About dataInfo.txt file: 
