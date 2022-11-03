## Dataset

This folder is the root data folder containing processed dataset, which needs to be downloaded and processed by the users. 




### Data Preprocessing

In our experiments, we conduct the data preprocessing steps on the two datasets as follows:


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




#### About dataInfo.txt File

As our experiments are performed on 2d sagittal slices, we utilize a dataInfo.txt file to save the index range of sagittal slices containing foreground (i.e., brain in our datasets), and the training processes are only conducted over these slices containing foreground.

Each row in the dataInfo.txt file corresponds to one subject. Take the first row for example:

    1 240 134 134 53 186

From left to right, each number represents:

    1:    the subject id
    240:  the total number of sagittal slices
    134:  the total number of sagittal slices containing foreground
    134:  the final number of sagittal slices containing foreground
    53:   the index of first sagittal slice containing foreground
    186:  the index of last sagittal slice containing foreground
    
The users could produce your own dataInfo.txt file based on your preprocessed dataset.
