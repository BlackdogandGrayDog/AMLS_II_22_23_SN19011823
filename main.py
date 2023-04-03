#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 23:13:44 2023

"""
print("Importing all modules...")
# Import necessary libraries and modules
import dataset_preprocessing.crop_image_preprocessing as cip
import initial_model_selection.initial_selection_training as ist
import gan_based_model.gan_construction_training as gct
import model_tuning.architecture_tuning as at
import model_tuning.hyperparameter_tuning as ht
import final_construction_testing.final_testing as ft
print("Importing successful...")

# Define paths for the dataset
hr_folder_path = 'Dataset/DIV2K_train_HR'
lr_x2_folder_path = 'Dataset/DIV2K_train_LR_bicubic/X2'
lr_x3_folder_path = 'Dataset/DIV2K_train_LR_bicubic/X3'
lr_x4_folder_path = 'Dataset/DIV2K_train_LR_bicubic/X4'

hr_val_path = 'Dataset/DIV2K_valid_HR'
lr_x2_val_path = 'Dataset/DIV2K_valid_LR_bicubic/X2'
lr_x3_val_path = 'Dataset/DIV2K_valid_LR_bicubic/X3'
lr_x4_val_path = 'Dataset/DIV2K_valid_LR_bicubic/X4'

lr_x2_folder_path_un = 'Dataset/DIV2K_train_LR_unknown/X2'
lr_x3_folder_path_un = 'Dataset/DIV2K_train_LR_unknown/X3'
lr_x4_folder_path_un = 'Dataset/DIV2K_train_LR_unknown/X4'

lr_x2_val_path_un = 'Dataset/DIV2K_valid_LR_unknown/X2'
lr_x3_val_path_un = 'Dataset/DIV2K_valid_LR_unknown/X3'
lr_x4_val_path_un = 'Dataset/DIV2K_valid_LR_unknown/X4'

# Perform dataset preprocessing and cropping
print("Preprocessing and cropping images...")
hr_image_paths, lr_x2_image_paths, lr_x3_image_paths, lr_x4_image_paths = cip.plot_cropped_images(hr_folder_path, lr_x2_folder_path, lr_x3_folder_path, lr_x4_folder_path, image_number=92)
hr_crop_image, lr_x2_crop_image,  lr_x3_crop_image, lr_x4_crop_image = cip.read_crop_images(hr_image_paths, lr_x2_image_paths, lr_x3_image_paths, lr_x4_image_paths)

hr_image_paths, lr_x2_image_paths, lr_x3_image_paths, lr_x4_image_paths = cip.plot_cropped_images(hr_val_path, lr_x2_val_path, lr_x3_val_path, lr_x4_val_path, image_number=87)
hr_val_image, lr_x2_val_image, lr_x3_val_image, lr_x4_val_image = cip.read_crop_images(hr_image_paths, lr_x2_image_paths, lr_x3_image_paths, lr_x4_image_paths)

hr_image_paths_un, lr_x2_image_paths_un, lr_x3_image_paths_un, lr_x4_image_paths_un = cip.plot_cropped_images(hr_folder_path, lr_x2_folder_path_un, lr_x3_folder_path_un, lr_x4_folder_path_un, image_number=92)
hr_crop_image_un, lr_x2_crop_image_un,  lr_x3_crop_image_un, lr_x4_crop_image_un = cip.read_crop_images(hr_image_paths_un, lr_x2_image_paths_un, lr_x3_image_paths_un, lr_x4_image_paths_un)

hr_image_paths_unv, lr_x2_image_paths_unv, lr_x3_image_paths_unv, lr_x4_image_paths_unv = cip.plot_cropped_images(hr_val_path, lr_x2_val_path_un, lr_x3_val_path_un, lr_x4_val_path_un, image_number=87)
hr_val_image_un, lr_x2_val_image_un, lr_x3_val_image_un, lr_x4_val_image_un = cip.read_crop_images(hr_image_paths_unv, lr_x2_image_paths_unv, lr_x3_image_paths_unv, lr_x4_image_paths_unv)

# Perform initial training on the dataset
print("Performing initial training...")
ist.initial_train(lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image)

# Train and plot the SRGAN model
print("Training and plotting SRGAN...")
gct.train_and_plot_srgan(lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image, scale_factor=4, epochs=20, batch_size=1, selection=True)

# Train and evaluate GAN models with different architectures
print("Training and evaluating GAN models with different architectures...")
at.train_and_evaluate_gan_models(lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image)

# Perform hyperparameter tuning for GAN models
print("Performing hyperparameter tuning for GAN models...")
ht.gan_tuning(lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image, max_trials=10, search_epochs=20, train_epochs=20, batch_size=1)

# Perform batch size tuning for GAN models
print("Performing batch size tuning for GAN models...")
ht.batch_size_tuning(lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image)

# Train and evaluate all models using the final construction and testing process
print("Training and evaluating all models using the final construction and testing process...")
ft.train_and_evaluate_all_models(hr_crop_image, hr_val_image, hr_crop_image_un, hr_val_image_un)

print("All processes completed successfully.")

