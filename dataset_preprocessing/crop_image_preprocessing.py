#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:08:19 2023
"""

"""
This script provides functions for cropping and reading high-resolution images and their corresponding low-resolution
images at different scaling factors (x2, x3, and x4) for use in a super-resolution GAN.
"""

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt


#%%
def plot_cropped_images(hr_folder_path, lr_x2_folder_path, lr_x3_folder_path, lr_x4_folder_path, image_number=80, crop_size=300):
    """
    This function plots a randomly selected cropped area from a high-resolution image and its corresponding 
    low-resolution images at different scaling factors (x2, x3, and x4). It displays the original image with the
    cropped area highlighted, followed by the cropped images at different resolutions.

    Args:
        hr_folder_path (str): Path to the folder containing high-resolution images.
        lr_x2_folder_path (str): Path to the folder containing low-resolution images at x2 scaling factor.
        lr_x3_folder_path (str): Path to the folder containing low-resolution images at x3 scaling factor.
        lr_x4_folder_path (str): Path to the folder containing low-resolution images at x4 scaling factor.
        image_number (int): Index of the image to be used for cropping and displaying. Default is 80.
        crop_size (int): Size of the square cropped area. Default is 300.

    Returns:
        tuple: A tuple containing four lists of sorted image paths for the high-resolution images and low-resolution images at x2, x3, and x4 scaling factors.
    """

    # Get sorted lists of image paths from the specified folders
    hr_image_paths = sorted(glob.glob(os.path.join(hr_folder_path, '*.png')))
    lr_x2_image_paths = sorted(glob.glob(os.path.join(lr_x2_folder_path, '*.png')))
    lr_x3_image_paths = sorted(glob.glob(os.path.join(lr_x3_folder_path, '*.png')))
    lr_x4_image_paths = sorted(glob.glob(os.path.join(lr_x4_folder_path, '*.png')))
    
    # Select the image paths at the specified index (image_number)
    hr_image_path = hr_image_paths[image_number]
    lr_x2_image_path = lr_x2_image_paths[image_number]
    lr_x3_image_path = lr_x3_image_paths[image_number]
    lr_x4_image_path = lr_x4_image_paths[image_number]

    # Read and convert images to RGB
    hr_image = cv2.cvtColor(cv2.imread(hr_image_path), cv2.COLOR_BGR2RGB)
    lr_x2_image = cv2.cvtColor(cv2.imread(lr_x2_image_path), cv2.COLOR_BGR2RGB)
    lr_x3_image = cv2.cvtColor(cv2.imread(lr_x3_image_path), cv2.COLOR_BGR2RGB)
    lr_x4_image = cv2.cvtColor(cv2.imread(lr_x4_image_path), cv2.COLOR_BGR2RGB)

    # Randomly select the top-left corner of the cropped area
    y = np.random.randint(0, hr_image.shape[0] - crop_size)
    x = np.random.randint(0, hr_image.shape[1] - crop_size)

    # Crop and normalize images
    hr_crop = hr_image[y:y+crop_size, x:x+crop_size, :]/255.0
    lr_x2_crop = lr_x2_image[y//2:y//2+crop_size//2, x//2:x//2+crop_size//2, :]/255.0
    lr_x3_crop = lr_x3_image[y//3:y//3+crop_size//3, x//3:x//3+crop_size//3, :]/255.0
    lr_x4_crop = lr_x4_image[y//4:y//4+crop_size//4, x//4:x//4+crop_size//4, :]/255.0

    # Plot the original image with the cropped area highlighted
    plt.figure(figsize=(8, 8))
    plt.imshow(hr_image)
    plt.gca().add_patch(plt.Rectangle((x, y), crop_size, crop_size, edgecolor='r', facecolor='none'))
    plt.title("Original Image with Cropped Area")
    plt.show()

    # Plot the cropped images at different resolutions
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    ax[0, 0].imshow(hr_crop)
    ax[0, 0].set_title('HR Crop')

    ax[0, 1].imshow(lr_x2_crop)
    ax[0, 1].set_title('x2 Crop')

    ax[1, 0].imshow(lr_x3_crop)
    ax[1, 0].set_title('x3 Crop')

    ax[1, 1].imshow(lr_x4_crop)
    ax[1, 1].set_title('x4 Crop')

    plt.show()

    return hr_image_paths, lr_x2_image_paths, lr_x3_image_paths, lr_x4_image_paths

#%%
def read_crop_images(hr_image_paths, lr_x2_image_paths, lr_x3_image_paths, lr_x4_image_paths, crop_size=300):
    """
    This function reads and crops high-resolution images and their corresponding low-resolution images at
    different scaling factors (x2, x3, and x4). It returns numpy arrays containing the cropped image data.

    Args:
        hr_image_paths (list): List of file paths for high-resolution images.
        lr_x2_image_paths (list): List of file paths for low-resolution images at x2 scaling factor.
        lr_x3_image_paths (list): List of file paths for low-resolution images at x3 scaling factor.
        lr_x4_image_paths (list): List of file paths for low-resolution images at x4 scaling factor.
        crop_size (int): Size of the square cropped area. Default is 300.

    Returns:
        tuple: A tuple containing four numpy arrays with the cropped image data for the high-resolution images and low-resolution images at x2, x3, and x4 scaling factors.
    """
    
    # Initialize empty lists to store cropped image data
    hr_crop_image, lr_x2_crop_image, lr_x3_crop_image, lr_x4_crop_image = [], [], [], []

    # Iterate over image paths
    for hr_image_path, lr_x2_image_path, lr_x3_image_path, lr_x4_image_path in zip(hr_image_paths, lr_x2_image_paths, lr_x3_image_paths, lr_x4_image_paths):
      
        # Read the images
        hr_image = cv2.imread(hr_image_path)
        lr_x2_image = cv2.imread(lr_x2_image_path)
        lr_x3_image = cv2.imread(lr_x3_image_path)
        lr_x4_image = cv2.imread(lr_x4_image_path)
      
        # Generate random coordinates for cropping
        y = np.random.randint(0, hr_image.shape[0] - crop_size)
        x = np.random.randint(0, hr_image.shape[1] - crop_size)

        # Crop the images
        hr_crop = hr_image[y:y+crop_size, x:x+crop_size, :]/255.0
        lr_x2_crop = lr_x2_image[y//2:y//2+crop_size//2, x//2:x//2+crop_size//2, :]/255.0
        lr_x3_crop = lr_x3_image[y//3:y//3+crop_size//3, x//3:x//3+crop_size//3, :]/255.0
        lr_x4_crop = lr_x4_image[y//4:y//4+crop_size//4, x//4:x//4+crop_size//4, :]/255.0

        # Append the cropped images to their respective lists
        hr_crop_image.append(hr_crop)
        lr_x2_crop_image.append(lr_x2_crop)
        lr_x3_crop_image.append(lr_x3_crop)
        lr_x4_crop_image.append(lr_x4_crop)

    # Convert the lists of cropped images to numpy arrays and return
    return np.array(hr_crop_image), np.array(lr_x2_crop_image), np.array(lr_x3_crop_image), np.array(lr_x4_crop_image)


#%%
# lr_x2_folder_path = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0135 AMLS II/Assignment/AMLSII_22-23_Assignment_SN19011823/Dataset/DIV2K_train_LR_bicubic/X2'
# lr_x3_folder_path = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0135 AMLS II/Assignment/AMLSII_22-23_Assignment_SN19011823/Dataset/DIV2K_train_LR_bicubic/X3'
# lr_x4_folder_path = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0135 AMLS II/Assignment/AMLSII_22-23_Assignment_SN19011823/Dataset/DIV2K_train_LR_bicubic/X4'
# hr_folder_path = '/Users/ericwei/Documents/UCL/Postgraduate/ELEC0135 AMLS II/Assignment/Assignment_AMLS/Dataset/DIV2K_train_HR'

# hr_image_paths, lr_x2_image_paths, lr_x3_image_paths, lr_x4_image_paths = plot_cropped_images(hr_folder_path, lr_x2_folder_path, lr_x3_folder_path, lr_x4_folder_path, image_number=92, crop_size=256)

# hr_crop_image, lr_x2_crop_image,  lr_x3_crop_image, lr_x4_crop_image = read_crop_images(hr_image_paths, lr_x2_image_paths, lr_x3_image_paths, lr_x4_image_paths)

#%%
