## Usage

This folder contains the code of Hyper-GAN using conditional instance normalization strategy. The structure of this folder is:

    ours_model_cin\      : code of Hyper-GAN using conditional instance normalization strategy
           |-- main.py         : main function
           |-- model.py        : code of building model, and train/valid/test
           |-- module.py       : code of defining networks
           |-- ops.py          : code of defining basic components
           |-- utils.py        : code of loading train and test data


### Training

After setting the phase in main.py as "train", our code can be trained using the following commond:

    python main.py

If you want to continue train the model, you could uncommond the continue_training codes in train function within model.py, and then run the commond above.


### Validation

Before starting the validation process, you need to set the phase in main.py as "valid". (Maybe also need to modify the information about valid set and epoch in valid function within model.py.)
Then, the validation process can be conducted using the following commond:

    python main.py
    
After generating the validation results, you could select the optimal epoch_id based on the performance on validation set.


### Test

Before starting the test process, you need to set the phase in main.py as "test" and set the epoch as the selected optimal epoch_id in test function within model.py.
Then, you can generate the test results using the following commond:

    python main.py


## Citation
If you use this code for your research, please cite our paper:
> @inproceedings{yang2021unified,
> <br> title={A Unified Hyper-GAN Model for Unpaired Multi-contrast MR Image Translation},
> <br> author={Yang, Heran and Sun, Jian and Yang, Liwei and Xu, Zongben},
> <br> booktitle={MICCAI},
> <br> pages={127--137},
> <br> year={2021}}
