#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:48:59 2023

"""

'''
This script streamlines the initial model selection, training, and performance comparison process for image super-resolution tasks. 
It includes functions for training SRCNN, SRResNet, AutoEncoder, compiling and training these models, and visualising generated images and performance metrics.
'''
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import matplotlib.pyplot as plt
import initial_model_construction as im
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
#%%
def psnr_metric(y_true, y_pred):
    """
    Computes the peak signal-to-noise ratio (PSNR) between the ground truth and predicted images.

    Args:
        y_true (tf.Tensor): Ground truth images.
        y_pred (tf.Tensor): Predicted images.

    Returns:
        tf.Tensor: PSNR value between the ground truth and predicted images.
    """
    max_pixel = 1.0 # Define the maximum pixel value for normalization
    psnr_value = tf.image.psnr(y_true, y_pred, max_val=max_pixel) # Calculate PSNR using TensorFlow's psnr() function
    return psnr_value

#%%
def ssim_metric(y_true, y_pred):
    """
    Computes the structural similarity index measure (SSIM) between the ground truth and predicted images.

    Args:
        y_true (tf.Tensor): Ground truth images.
        y_pred (tf.Tensor): Predicted images.

    Returns:
        tf.Tensor: SSIM value between the ground truth and predicted images.
    """
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


#%%
def compile_and_train(model, lr_images, hr_images, epochs, batch_size):
    """
    Compiles and trains the given model on low-resolution and high-resolution images.

    Args:
        model (keras.Model): The model to be trained.
        lr_images (np.array): Low-resolution images.
        hr_images (np.array): High-resolution images.
        epochs (int): Number of training epochs.
        batch_size (int): Size of the training batches.

    Returns:
        keras.callbacks.History: A history object containing training metrics and losses.
    """
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mse', 'accuracy', psnr_metric, ssim_metric])
    history = model.fit(x=lr_images, y=hr_images, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return history


#%%
def plot_generated_image(model, image_num, lr_x4_val_image, hr_val_image, resize=False):
    """
    Plots a comparison between the original high-resolution (HR) image,
    the generated high-resolution image from the model, and the original low-resolution (LR) image.

    Args:
        model (tensorflow.keras.Model): Trained GAN generator model used to generate the high-resolution image.
        image_num (int): Index of the test image to be used for comparison.
        resize (bool, optional): If True, the low-resolution image will be resized using bicubic interpolation
                                 before passing it to the model (default is False).

    Returns:
        tuple: Generated high-resolution image, real high-resolution image, and the original low-resolution image.
    """
    # Resize the LR image using bicubic interpolation if resize is set to True
    if resize:
        test_lr = cv2.resize(lr_x4_val_image[image_num], None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    else:
        test_lr = lr_x4_val_image[image_num]

    # Prepare the LR image for the model and generate the HR image
    test_lr = np.expand_dims(test_lr, axis=0)
    gen_hr = model.predict(test_lr)[0]
    gen_hr = (gen_hr * 255).clip(0, 255).astype(np.uint8)

    # Get the real HR image and convert it to the proper format
    real_hr = (hr_val_image[image_num] * 255).clip(0, 255).astype(np.uint8)
    
    # Resize gen_lr to the same size as gen_hr and real_hr
    gen_lr = (lr_x4_val_image[image_num] * 255).clip(0, 255).astype(np.uint8)
    gen_lr = cv2.resize(gen_lr, (gen_hr.shape[1], gen_hr.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Plot the original HR image, generated HR image, and original LR image for comparison
    fig, axs = plt.subplots(1, 3, figsize=(20, 20))
    axs[0].imshow(cv2.cvtColor(real_hr, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original HR Image', fontsize=20, fontweight='bold')
    axs[0].tick_params(axis='both', labelsize=15)
    axs[1].imshow(cv2.cvtColor(gen_hr, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Generated HR Image', fontsize=20, fontweight='bold')
    axs[1].tick_params(axis='both', labelsize=15)
    axs[2].imshow(cv2.cvtColor(gen_lr, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Original LR Image', fontsize=20, fontweight='bold')
    axs[2].tick_params(axis='both', labelsize=20)

    plt.show()

    return gen_hr, real_hr, gen_lr

#%%
def plot_metrics_initial(history):
    """
    Plots the training and validation metrics (loss, accuracy, PSNR, SSIM) over the training epochs.

    Args:
        history (keras.callbacks.History): A history object containing training metrics and losses.
    """
    plt.figure(figsize=(30, 20))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epoch', fontsize=20, fontweight='bold')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epoch', fontsize=20, fontweight='bold')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Plot PSNR
    plt.subplot(2, 2, 3)
    plt.plot(history.history['psnr_metric'], label='Training PSNR')
    plt.plot(history.history['val_psnr_metric'], label='Validation PSNR')
    plt.title('PSNR vs. Epoch', fontsize=20, fontweight='bold')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('PSNR', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Plot SSIM
    plt.subplot(2, 2, 4)
    plt.plot(history.history['ssim_metric'], label='Training SSIM')
    plt.plot(history.history['val_ssim_metric'], label='Validation SSIM')
    plt.title('SSIM vs. Epoch', fontsize=20, fontweight='bold')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('SSIM', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()
    
#%%

def train_and_plot_model(model_name, lr_shape, scale_factor, lr_images, hr_images, lr_val_images, hr_val_images, epochs, batch_size, resize_flag= False):
    """
    Trains the specified model, plots the results, and calculates PSNR and SSIM values.

    Args:
        model_name (str): Name of the model to train ('SRCNN', 'AutoEncoder', 'Combined', 'SRResNet').
        lr_shape (tuple): Shape of the low-resolution input image.
        scale_factor (int): Scaling factor for upscaling.
        lr_images (np.array): Array of low-resolution images.
        hr_images (np.array): Array of high-resolution images.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.

    Returns:
        None
    """

    if model_name == 'SRCNN':
        model = im.srcnn(lr_shape, scale_factor)
    elif model_name == 'AutoEncoder':
        model = im.autoencoder()
        lr_images = im.resize_images(lr_images, scale_factor)
    elif model_name == 'Combined':
        model = im.combined_srcnn_autoencoder(lr_shape, scale_factor)
    elif model_name == 'SRResNet':
        model = im.srresnet(lr_shape, scale_factor)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    history = compile_and_train(model, lr_images, hr_images, epochs, batch_size)
    model.save(f'{model_name}_model.h5')

    # Plot the generated image
    plot_metrics_initial(history)
    gen_hr, real_hr, gen_lr = plot_generated_image(model, 1, lr_val_images, hr_val_images, resize = resize_flag)
    gen_hr = np.array(gen_hr)
    real_hr = np.array(real_hr)
    # Calculate PSNR
    psnr_value = psnr(real_hr, gen_hr, data_range=real_hr.max() - real_hr.min())

    # Calculate SSIM
    ssim_value = ssim(real_hr, gen_hr, multichannel=True, data_range=real_hr.max() - real_hr.min(), win_size = 3)
    print(f"Average PSNR: {psnr_value}")
    print(f"Average SSIM: {ssim_value}")

#%%
def initial_train(lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image):
    """
    Train and plot all four models: SRCNN, AutoEncoder, Combined, and SRResNet.

    Args:
        lr_x4_crop_image (numpy.ndarray): Low-resolution images with a 4x scale factor.
        hr_crop_image (numpy.ndarray): High-resolution images.

    Returns:
        None
    """
    # Set common training parameters
    lr_shape = (75, 75, 3)
    scale_factor = 4
    epochs = 100
    batch_size = 4

    # Training and plotting for SRCNN
    model_name = 'SRCNN'
    train_and_plot_model(model_name, lr_shape, scale_factor, lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image, epochs, batch_size)

    # Training and plotting for AutoEncoder
    model_name = 'AutoEncoder'
    train_and_plot_model(model_name, lr_shape, scale_factor, lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image, epochs, batch_size, resize_flag = True)

    # Training and plotting for Combined Model
    model_name = 'Combined'
    train_and_plot_model(model_name, lr_shape, scale_factor, lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image, epochs, batch_size)

    # Training and plotting for SRResNet
    model_name = 'SRResNet'
    train_and_plot_model(model_name, lr_shape, scale_factor, lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image, epochs, batch_size)