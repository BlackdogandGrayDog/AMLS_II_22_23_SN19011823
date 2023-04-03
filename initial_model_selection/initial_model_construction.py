#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:52:00 2023

"""

"""
This script contains multiple deep learning model architectures for image super-resolution tasks. The main goal of these models is to upscale low-resolution images to high-resolution images while maintaining or improving image quality. This file includes the following architectures:

SRCNN (Super-Resolution Convolutional Neural Network)
SRResNet (Super-Resolution Residual Network)
Autoencoder-based Super-Resolution
Combined SRCNN and Autoencoder
"""

import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, PReLU, UpSampling2D, MaxPool2D, add
from tensorflow.keras.models import Model
#%%
def srcnn(lr_shape, scale_factor):
    """
    Defines the Super-Resolution Convolutional Neural Network (SRCNN) model architecture.

    Args:
        lr_shape (tuple): The shape of the input low-resolution image.
        scale_factor (int): The factor by which the low-resolution image will be upscaled.

    Returns:
        tensorflow.keras.Model: The SRCNN model.
    """
    # Define the input layer with the low-resolution image shape
    input_layer = Input(shape=lr_shape)

    # Upsample the low-resolution image using bilinear interpolation
    upsampled = UpSampling2D(size=scale_factor, interpolation='bilinear')(input_layer)
    
    # Apply a convolutional layer with 64 filters, 9x9 kernel size, ReLU activation, and same padding
    x = Conv2D(64, kernel_size=9, activation='relu', padding='same')(upsampled)
    # Apply a convolutional layer with 32 filters, 1x1 kernel size, ReLU activation, and same padding
    x = Conv2D(32, kernel_size=1, activation='relu', padding='same')(x)
    # Apply a convolutional layer with 3 filters, 5x5 kernel size, and same padding to obtain the final high-resolution image
    x = Conv2D(3, kernel_size=5, padding='same')(x)

    # Create and return the SRCNN model
    return Model(input_layer, x)

#%%
def res_block(x, filters):
    """
    Defines a residual block, a commonly used building block in deep learning architectures.

    Args:
        x (tensor): The input tensor to the residual block.
        filters (int): The number of filters for the convolutional layers.

    Returns:
        tensor: The output tensor of the residual block.
    """
    res = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    res = BatchNormalization()(res)
    res = PReLU()(res)
    res = Conv2D(filters, kernel_size=3, strides=1, padding='same')(res)
    res = BatchNormalization()(res)
    return Add()([x, res])

def upsampling_block(x, num_filters):
    """
    Defines an upsampling block that increases the spatial dimensions of the input tensor.

    Args:
        x (tensor): The input tensor to the upsampling block.
        num_filters (int): The number of filters for the convolutional layer.

    Returns:
        tensor: The output tensor of the upsampling block.
    """
    x = Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
    x = PReLU()(x)
    x = UpSampling2D()(x)
    return x

def srresnet(input_shape, num_res_blocks=16, num_filters=64):
    """
    Defines the SRResNet model architecture, a deep learning model for image super-resolution.

    Args:
        input_shape (tuple): The shape of the input low-resolution image.
        num_res_blocks (int): The number of residual blocks in the model (default: 16).
        num_filters (int): The number of filters for the convolutional layers (default: 64).

    Returns:
        tensorflow.keras.Model: The SRResNet model.
    """
    inp = Input(input_shape)
    x = Conv2D(num_filters, kernel_size=9, strides=1, padding='same')(inp)
    x = skip = PReLU()(x)
    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)
    x = Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, skip])

    # Upsampling blocks
    x = upsampling_block(x, 256)
    x = upsampling_block(x, 256)

    x = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(x)
    return Model(inputs=inp, outputs=x)

#%%
def autoencoder():
    """
    Defines an autoencoder model for image super-resolution.
    The model has an encoder-decoder architecture that extracts features from low-resolution images
    and reconstructs a high-resolution image from these features.
    """

    # Input layer for 300x300 images with 3 channels
    input_layer = Input(shape=(300, 300, 3))

    # Encoder
    # First convolutional block with 64 filters and ReLU activation
    l1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(input_layer)
    l2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l1)
    # Max-pooling layer to reduce spatial dimensions
    l3 = MaxPool2D(padding='same')(l2)

    # Second convolutional block with 128 filters and ReLU activation
    l4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l3)
    l5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l4)
    # Max-pooling layer to reduce spatial dimensions
    l6 = MaxPool2D(padding='same')(l4)

    # Third convolutional block with 256 filters and ReLU activation
    l7 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l6)

    # Decoder
    # Upsampling layer to increase spatial dimensions
    l8 = UpSampling2D()(l7)
    # First deconvolutional block with 128 filters and ReLU activation
    l9 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu')(l9)

    # Adding skip connection from encoder's second convolutional block
    l11 = add([l5, l10])

    # Upsampling layer to increase spatial dimensions
    l12 = UpSampling2D()(l11)
    # Second deconvolutional block with 64 filters and ReLU activation
    l13 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l12)
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu')(l13)

    # Adding skip connection from encoder's first convolutional block
    l15 = add([l14, l2])

    # Output layer with 3 filters for RGB channels and ReLU activation
    decoded_image = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l15)

    return Model(input_layer, decoded_image)


# Function to resize low-resolution images to the high-resolution size using linear interpolation
def resize_images(lr_images, scale_factor):
    """
    Resizes the low-resolution images to the high-resolution size using linear interpolation.

    Args:
        lr_images (numpy array): A batch of low-resolution images.
        scale_factor (int): The scaling factor for resizing the images.

    Returns:
        numpy array: A batch of resized images.
    """
    hr_images_resized = []

    # Loop through the low-resolution images
    for lr_img in lr_images:
        # Resize the low-resolution image using linear interpolation
        hr_img_resized = cv2.resize(lr_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        # Append the resized image to the list
        hr_images_resized.append(hr_img_resized)

    # Convert the list of resized images to a numpy array
    return np.array(hr_images_resized)

#%%
def combined_srcnn_autoencoder(lr_shape, scale_factor):
    """
    Defines the combined SRCNN and Autoencoder architecture for image super-resolution.

    Args:
        lr_shape (tuple): The shape of the low-resolution input images.
        scale_factor (int): The scaling factor for resizing the images.

    Returns:
        Model: A Keras model with the combined architecture of SRCNN and Autoencoder.
    """
    input_layer = Input(shape=lr_shape)

    # SRCNN
    # Upsampling low-resolution image using bilinear interpolation
    upsampled = UpSampling2D(size=scale_factor, interpolation='bilinear')(input_layer)
    x = Conv2D(64, kernel_size=9, activation='relu', padding='same')(upsampled)
    x = Conv2D(32, kernel_size=1, activation='relu', padding='same')(x)
    srcnn_output = Conv2D(3, kernel_size=5, padding='same')(x)

    # Autoencoder (Encoder)
    l1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(srcnn_output)
    l2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l1)
    l3 = MaxPool2D(padding='same')(l2)

    l4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l3)
    l5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l4)
    l6 = MaxPool2D(padding='same')(l4)

    l7 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l6)

    # Autoencoder (Decoder)
    l8 = UpSampling2D()(l7)
    l9 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu')(l9)

    l11 = add([l5, l10])

    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l12)
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu')(l13)

    l15 = add([l14, l2])

    # Output high-resolution image
    decoded_image = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l15)
    
    return Model(input_layer, decoded_image)