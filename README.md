# GAN Models: DCGAN and Conditional DCGAN with Auxiliary Classifier

## Overview

This repository contains implementations of two GAN models:

1. **DCGAN** on the Cats dataset (64x64)
2. **Conditional DCGAN (cDCGAN)** on the MNIST-4 dataset (32x32)

Note: The cDCGAN is currently in the development stage.

## Generating Random Cat Images

To generate random cat images using the pre-trained DCGAN model, run the following command:

```bash
cd DCGAN
python inference.py --checkpoint_file_location <path_to_checkpoint> --num_images <number_of_images> --save_dir <path_to_save_directory> --device <use_gpu_if_present>
```

## Generating Images using cDCGAN from classes {airplane, automobile, horse, ship}

To generate images conditioned on a class using the pre-trained cDCGAN model, run the following command:

```bash
cd cDCGAN
python inference.py --class_name <class number(0,1,2,3> --checkpoint_file_location <path_to_checkpoint> --num_images <number_of_images> --save_dir <path_to_save_directory> --device <use_gpu_if_present>
```

Training commands and files will be uploaded later
