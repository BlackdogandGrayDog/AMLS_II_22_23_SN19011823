#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 00:41:30 2023

"""

'''
This script provides a comprehensive workflow for constructing, training, and evaluating Generative Adversarial Networks (GANs) for image super-resolution tasks. 
It includes functions for building the SRGAN architecture with optional VGG19 integration, training the generator and discriminator models, 
and visualizing the generated high-resolution images along with performance metrics. 
'''
import sys
sys.path.append('../initial_model_selection')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D, MaxPool2D, add, Multiply, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from tensorflow.keras.applications.vgg19 import VGG19
import initial_selection_training as ist
#%%

def GAN_generator(lr_shape, scale_factor):
    """
    Creates a generator model for a Generative Adversarial Network (GAN) that combines the SRCNN, Self-Attention Mechanism, and Autoencoder.

    Args:
        lr_shape (tuple): Shape of the low-resolution input image.
        scale_factor (int): Scaling factor for upscaling.

    Returns:
        keras.Model: GAN generator model.
    """
    # Input layer for the low-resolution image
    input_layer = Input(shape=lr_shape)

    # SRCNN: Super-Resolution Convolutional Neural Network
    # Upsampling and initial convolutional layers
    upsampled = UpSampling2D(size=scale_factor, interpolation='bilinear')(input_layer)
    x = Conv2D(64, kernel_size=9, activation='relu', padding='same')(upsampled)
    x = Conv2D(32, kernel_size=1, activation='relu', padding='same')(x)
    x = Conv2D(3, kernel_size=5, padding='same')(x)
    srcnn_output = add([upsampled, x])

    # Self-Attention Mechanism
    # Apply global average pooling and reshape for the attention mechanism
    channels = 3
    attention = GlobalAveragePooling2D()(srcnn_output)
    attention = Reshape((1, 1, channels))(attention)
    # Dense layers to learn the attention weights
    attention = Dense(channels // 2, activation='selu', use_bias=False, kernel_initializer='he_uniform')(attention)
    attention = Dense(channels, activation='selu', use_bias=False, kernel_initializer='he_uniform')(attention)
    attention = Activation('sigmoid')(attention)
    # Multiply the attention weights with the SRCNN output
    srcnn_output = Multiply()([srcnn_output, attention])

    # Autoencoder (Encoder)
    # Convolutional layers and max-pooling for encoding
    l1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(srcnn_output)
    l2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l1)
    l3 = MaxPool2D(padding='same')(l2)
    l4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l3)
    l5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l4)
    l6 = MaxPool2D(padding='same')(l4)
    l7 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l6)

    # Autoencoder (Decoder)
    # Upsampling and convolutional layers for decoding
    l8 = UpSampling2D()(l7)
    l9 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu')(l9)
    l11 = add([l5, l10])
    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l12)
    
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu')(l13)
    l15 = add([l14, l2])
    
    # Decoder output: reconstructed high-resolution image
    decoded_image = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l15)
    
    # Combine the outputs from SRCNN, Autoencoder, and upsampled input
    decoded_output = add([srcnn_output, decoded_image, upsampled])
    
    # Second Self-Attention Mechanism (applied to the combined output)
    channels = 3
    attention = GlobalAveragePooling2D()(decoded_output)
    attention = Reshape((1, 1, channels))(attention)
    attention = Dense(channels // 2, activation='selu', use_bias=False, kernel_initializer='he_uniform')(attention)
    attention = Dense(channels, activation='selu', use_bias=False, kernel_initializer='he_uniform')(attention)
    attention = Activation('sigmoid')(attention)
    
    # Multiply the attention weights with the combined output
    output = Multiply()([decoded_output, attention])
    
    # Return the final GAN generator model
    return Model(input_layer, output)

#%%
def discriminator_block(input_img, num_filters, strides):
    """
    Create a discriminator block with Conv2D, BatchNormalization, and LeakyReLU layers.
    
    Args:
        input_img (tensor): The input tensor for the block.
        num_filters (int): Number of filters for the Conv2D layer.
        strides (int): Stride value for the Conv2D layer.
        
    Returns:
        model (tensor): The output tensor of the block.
    """
    model = Conv2D(num_filters, (3,3), strides = strides, padding='same')(input_img)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    return model

#%%
def GAN_discriminator(hr_shape):
    """
    Create the GAN discriminator model with the specified high-resolution shape.
    
    Args:
        hr_shape (tuple): Shape of the high-resolution input image.
        
    Returns:
        discriminator (Model): The GAN discriminator model.
    """
    input_img = Input(shape=hr_shape)
    
    # Initial Conv2D layer with LeakyReLU activation
    model = Conv2D(64, (3,3), strides = 1, padding='same')(input_img)
    model = LeakyReLU(alpha=0.2)(model)
    
    # Add discriminator blocks
    model = discriminator_block(model, 64, 2)
    model = discriminator_block(model, 128, 1)
    model = discriminator_block(model, 128, 2)
    model = discriminator_block(model, 256, 1)
    model = discriminator_block(model, 256, 2)
    model = discriminator_block(model, 512, 1)
    model = discriminator_block(model, 512, 2)
    
    # Flatten and add Dense layers
    model = Flatten()(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dense(1, activation='sigmoid')(model)
    
    # Create the discriminator model
    discriminator = Model(input_img, model)
    
    return discriminator

#%%
def vgg19_block():
    """
    Creates a VGG19 model with pre-trained ImageNet weights and includes all layers up to block4_conv4.

    Returns:
        Model: The VGG19 model.
    """
    model = VGG19(weights="imagenet", include_top=False, input_shape=(300, 300, 3))
    return Model(inputs=model.inputs, outputs=model.layers[9].output)

#%%
def SRGAN_block(GAN_generator, GAN_discriminator, lr_shape, hr_shape):
    """
    Create the SRGAN model with generator and discriminator.
    
    Args:
        GAN_generator (Model): Generator model.
        GAN_discriminator (Model): Discriminator model.
        lr_shape (tuple): Shape of the low-resolution input image.
        hr_shape (tuple): Shape of the high-resolution input image.
        
    Returns:
        SRGAN_model (Model): The SRGAN model.
    """
    lr_img = Input(shape=lr_shape)
    hr_img = Input(shape=hr_shape)
    fake_img = GAN_generator(lr_img)

    GAN_discriminator.trainable = False
    pred_prob = GAN_discriminator(fake_img)

    SRGAN_model = Model(inputs=[lr_img, hr_img], outputs=[pred_prob, fake_img])

    return SRGAN_model

#%%
def SRGAN_block_vgg(generator, discriminator, vgg19, lr_shape):
    """
    Builds the SRGAN model with the VGG feature loss, including the generator, discriminator, and VGG19 models.

    Args:
        generator (Model): The generator model.
        discriminator (Model): The discriminator model.
        vgg19 (Model): The VGG19 model.
        lr_shape (tuple): The shape of the low-resolution input image.

    Returns:
        Model: The SRGAN model with the VGG feature loss.
    """
    input_layer = Input(shape=lr_shape)
    gen_output = generator(input_layer)
    disc_output = discriminator(gen_output)
    vgg_output = vgg19(gen_output)
    
    return Model(input_layer, [disc_output, vgg_output])

#%%
def train_srgan(gan_generator, gan_discriminator, srgan_model, lr_images, hr_images, val_split=0.1, epochs=20, batch_size=2):
    """
    Train the SRGAN model on the given dataset.
    
    Args:
        gan_generator (Model): Generator model.
        gan_discriminator (Model): Discriminator model.
        srgan_model (Model): SRGAN model.
        lr_images (numpy.ndarray): Low-resolution images.
        hr_images (numpy.ndarray): High-resolution images.
        val_split (float): Fraction of the dataset to be used as validation.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        
    Returns:
        Tuple: The trained generator, discriminator, and their loss and PSNR histories.
    """
    # Split the dataset into training and validation sets
    lr_train, lr_val, hr_train, hr_val = train_test_split(lr_images, hr_images, test_size=val_split, random_state=42)

    # Create batches for training and validation data
    train_lr_batches = []
    train_hr_batches = []
    val_lr_batches = []
    val_hr_batches = []

    for it in range(int(len(hr_train) / batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        train_hr_batches.append(hr_train[start_idx:end_idx])
        train_lr_batches.append(lr_train[start_idx:end_idx])

    for it in range(int(len(hr_val) / batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        val_hr_batches.append(hr_val[start_idx:end_idx])
        val_lr_batches.append(lr_val[start_idx:end_idx])

    # Initialize loss and PSNR histories
    d_loss_history = []
    g_loss_history = []
    psnr_history = []

    d_val_loss_history = []
    g_val_loss_history = []
    val_psnr_history = []

    # Training loop
    for e in range(epochs):
        fake_label = np.zeros((batch_size, 1))
        real_label = np.ones((batch_size, 1))

        g_losses = []
        d_losses = []
        psnrs = []

        g_val_losses = []
        d_val_losses = []
        val_psnrs = []

        # Iterate through training batches
        for b in tqdm(range(len(train_hr_batches))):
            lr_imgs = np.array(train_lr_batches[b])
            hr_imgs = np.array(train_hr_batches[b])

            # Generate fake high-resolution images
            fake_imgs = gan_generator.predict_on_batch(lr_imgs)

            # Train the discriminator
            gan_discriminator.trainable = True
            d_loss_gen = gan_discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss_real = gan_discriminator.train_on_batch(hr_imgs, real_label)
            gan_discriminator.trainable = False

            # Compute the discriminator loss
            d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

            # Train the SRGAN model
            g_loss, _, _ = srgan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, hr_imgs])

            # Append losses to the lists
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            # Compute the PSNR
            psnr_value = psnr(hr_imgs, fake_imgs, data_range=hr_imgs.max() - hr_imgs.min())
            psnrs.append(psnr_value)

        # Iterate through validation batches
        for b in tqdm(range(len(val_hr_batches))):
            lr_imgs = np.array(val_lr_batches[b])
            hr_imgs = np.array(val_hr_batches[b])

            # Generate fake high-resolution images
            fake_imgs = gan_generator.predict_on_batch(lr_imgs)

            # Test the discriminator
            d_loss_gen = gan_discriminator.test_on_batch(fake_imgs, fake_label)
            d_loss_real = gan_discriminator.test_on_batch(hr_imgs, real_label)

            # Compute the discriminator validation loss
            d_val_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
            g_val_loss, _, _ = srgan_model.test_on_batch([lr_imgs, hr_imgs], [real_label, hr_imgs])

            # Append validation losses to the lists
            d_val_losses.append(d_val_loss)
            g_val_losses.append(g_val_loss)

            # Compute the validation PSNR
            psnr_val_value = psnr(hr_imgs, fake_imgs, data_range=hr_imgs.max() - hr_imgs.min())
            val_psnrs.append(psnr_val_value)

        # Record the average losses and PSNR values for this epoch
        d_loss_history.append(np.mean(d_losses))
        g_loss_history.append(np.mean(g_losses))
        psnr_history.append(np.mean(psnrs))

        d_val_loss_history.append(np.mean(d_val_losses))
        g_val_loss_history.append(np.mean(g_val_losses))
        val_psnr_history.append(np.mean(val_psnrs))

        # Print the losses and PSNR values for this epoch
        print(f"epoch: {e+1}, g_loss: {g_loss_history[-1]}, d_loss: {d_loss_history[-1]}, PSNR: {psnr_history[-1]}, Val_g_loss: {g_val_loss_history[-1]}, Val_d_loss: {d_val_loss_history[-1]}, Val_PSNR: {val_psnr_history[-1]}")

        # Save the generator model every 10 epochs
        if (e+1) % 10 == 0:
            gan_generator.save("gen_e_" + str(e+1) + ".h5")

    # Return the trained models and their loss and PSNR histories
    return gan_generator, gan_discriminator, d_loss_history, g_loss_history, psnr_history, d_val_loss_history, g_val_loss_history, val_psnr_history

#%%
def train_srgan_vgg(gan_generator, gan_discriminator, srgan_model, vgg_network, lr_images, hr_images, val_split=0.1, epochs=30, batch_size=4):
    """
    Train the SRGAN model on the given dataset using the VGG19 loss.

    Args:
        gan_generator (Model): Generator model.
        gan_discriminator (Model): Discriminator model.
        srgan_model (Model): SRGAN model.
        vgg_network (Model): VGG19 model for feature extraction.
        lr_images (numpy.ndarray): Low-resolution images.
        hr_images (numpy.ndarray): High-resolution images.
        val_split (float): Fraction of the dataset to be used as validation.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        Tuple: The trained generator, discriminator, and their loss and PSNR histories.
    """

    # Split dataset into training and validation sets
    lr_train, lr_val, hr_train, hr_val = train_test_split(lr_images, hr_images, test_size=val_split, random_state=42)

    # Create batches of low-resolution and high-resolution images for training and validation
    train_lr_batches = []
    train_hr_batches = []
    val_lr_batches = []
    val_hr_batches = []

    for it in range(int(len(hr_train) / batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        train_hr_batches.append(hr_train[start_idx:end_idx])
        train_lr_batches.append(lr_train[start_idx:end_idx])

    for it in range(int(len(hr_val) / batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        val_hr_batches.append(hr_val[start_idx:end_idx])
        val_lr_batches.append(lr_val[start_idx:end_idx])

    # Initialize lists to store losses and PSNR values
    epoch_g_losses = []
    epoch_d_losses = []
    epoch_psnr_values = []
    val_g_losses = []
    val_d_losses = []
    val_psnr_values = []
    
    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
    
        g_losses = []
        d_losses = []
        val_g_loss = []
        val_d_loss = []
    
        # Train on batches of images
        for batch in tqdm(range(len(train_hr_batches))):
            lr_imgs = np.array(train_lr_batches[batch])
            hr_imgs = np.array(train_hr_batches[batch])
    
            fake_imgs = gan_generator.predict(lr_imgs)
    
            # Train discriminator on fake and real images
            gan_discriminator.trainable = True
            d_loss_gen = gan_discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
            d_loss_real = gan_discriminator.train_on_batch(hr_imgs, np.ones((batch_size, 1)))
            gan_discriminator.trainable = False
    
            # Compute discriminator loss as the mean of fake and real image losses
            d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
    
            # Extract VGG features from high-resolution images
            image_features = vgg_network.predict(hr_imgs)
    
            # Train generator using VGG loss and binary cross-entropy loss
            g_loss, _, _ = srgan_model.train_on_batch(lr_imgs, [np.ones((batch_size, 1)), image_features])
    
            d_losses.append(d_loss)
            g_losses.append(g_loss)
    
        # Evaluate on validation set
        for batch in range(len(val_hr_batches)):
            val_lr_imgs = np.array(val_lr_batches[batch])
            val_hr_imgs = np.array(val_hr_batches[batch])
    
            val_fake_imgs = gan_generator.predict(val_lr_imgs)
    
            val_d_loss_gen = gan_discriminator.evaluate(val_fake_imgs, np.zeros((batch_size, 1)), verbose=0)
            val_d_loss_real = gan_discriminator.evaluate(val_hr_imgs, np.ones((batch_size, 1)), verbose=0)
            val_d_loss.append(0.5 * np.add(val_d_loss_gen, val_d_loss_real))
    
            val_image_features = vgg_network.predict(val_hr_imgs)
            val_g_loss.append(srgan_model.evaluate(val_lr_imgs, [np.ones((batch_size, 1)), val_image_features], verbose=0)[0])
    
        # Compute mean losses and PSNR values
        g_loss = np.mean(g_losses)
        d_loss = np.mean(d_losses)
        val_g_loss_avg = np.mean(val_g_loss)
        val_d_loss_avg = np.mean(val_d_loss)
        psnr_value = np.mean([psnr(hr_img, fake_img, data_range=hr_img.max() - hr_img.min()) for hr_img, fake_img in zip(hr_imgs, fake_imgs)])
        val_psnr_value = np.mean([psnr(hr_img, fake_img, data_range=hr_img.max() - hr_img.min()) for hr_img, fake_img in zip(val_hr_imgs, val_fake_imgs)])
    
        # Append losses and PSNR values to history
        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss)
        epoch_psnr_values.append(psnr_value)
        val_g_losses.append(val_g_loss_avg)
        val_d_losses.append(val_d_loss_avg)
        val_psnr_values.append(val_psnr_value)
    
        # Print current losses and PSNR values
        print(f"Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}, PSNR: {psnr_value:.4f}")
        print(f"Validation Generator Loss: {val_g_loss_avg:.4f}, Validation Discriminator Loss: {val_d_loss_avg:.4f}, Validation PSNR: {val_psnr_value:.4f}")
    
        # Save generator every 10 epochs
        if (epoch+1) % 10 == 0:
            gan_generator.save("gen_vgg_e_" + str(epoch+1) + ".h5")
    
    # Return losses and PSNR values for all epochs
    return gan_generator, gan_discriminator, epoch_g_losses, epoch_d_losses, epoch_psnr_values, val_g_losses, val_d_losses, val_psnr_values

#%%
def plot_metrics(d_loss_history, g_loss_history, psnr_history, d_val_loss_history, g_val_loss_history, val_psnr_history):
    """
    Plot the training and validation loss and PSNR values over epochs for a GAN model.

    Args:
        d_loss_history (list): List of discriminator training loss values.
        g_loss_history (list): List of generator training loss values.
        psnr_history (list): List of PSNR values for the training set.
        d_val_loss_history (list): List of discriminator validation loss values.
        g_val_loss_history (list): List of generator validation loss values.
        val_psnr_history (list): List of PSNR values for the validation set.

    Returns:
        None
    """
    # Define the range of epochs
    epochs = range(1, len(d_loss_history) + 1)

    # Set the figure size and create subplots for loss and PSNR
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.subplot(132)
    plt.subplot(133)

    # Plot the training and validation loss in the first subplot
    plt.subplot(131)
    plt.plot(epochs, d_loss_history, label='Training D loss')
    plt.plot(epochs, d_val_loss_history, label='Validation D loss')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Discriminator Loss', fontsize=15)
    plt.title('Discriminator Loss vs Epochs', fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    # Plot the training and validation generator loss in the second subplot
    plt.subplot(132)
    plt.plot(epochs, g_loss_history, label='Training G loss')
    plt.plot(epochs, g_val_loss_history, label='Validation G loss')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Generator Loss', fontsize=15)
    plt.title('Generator Loss vs Epochs', fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    # Plot the training and validation PSNR in the third subplot
    plt.subplot(133)
    plt.plot(epochs, psnr_history, label='Train PSNR')
    plt.plot(epochs, val_psnr_history, label='Validation PSNR')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('PSNR', fontsize=15)
    plt.title('PSNR vs Epochs', fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()

#%%
def plot_metrics_vgg(epoch_g_losses, epoch_d_losses, epoch_psnr_values, val_g_losses, val_d_losses, val_psnr_values):
    """
    Plots the generator loss, discriminator loss, and PSNR values during training.

    Args:
        epoch_g_losses (list): List of generator losses for each epoch.
        epoch_d_losses (list): List of discriminator losses for each epoch.
        epoch_psnr_values (list): List of PSNR values for each epoch.
        val_g_losses (list): List of validation generator losses for each epoch.
        val_d_losses (list): List of validation discriminator losses for each epoch.
        val_psnr_values (list): List of validation PSNR values for each epoch.
    """
    
    epochs = range(1, len(epoch_g_losses) + 1)

    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.plot(epochs, epoch_d_losses, label='Training D loss')
    plt.plot(epochs, val_d_losses, label='Validation D loss')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Discriminator Loss', fontsize=15)
    plt.title('Discriminator Loss vs Epochs', fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    plt.subplot(132)
    plt.plot(epochs, epoch_g_losses, label='Training G loss')
    plt.plot(epochs, val_g_losses, label='Validation G loss')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Generator Loss', fontsize=15)
    plt.title('Generator Loss vs Epochs', fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    plt.subplot(133)
    plt.plot(epochs, epoch_psnr_values, label='Train PSNR')
    plt.plot(epochs, val_psnr_values, label='Validation PSNR')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('PSNR', fontsize=15)
    plt.title('PSNR vs Epochs', fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

#%%
def train_and_plot_srgan(lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image, scale_factor=4, epochs=20, batch_size=1, selection = True):
    # Define the non-VGG GAN
    lr_shape = lr_x4_crop_image[1].shape
    hr_shape = hr_crop_image[1].shape
    lr_images, hr_images = lr_x4_crop_image, hr_crop_image
    
    if selection:
        # Training and plotting for Combined Model
        model_name = 'Combined'
        ist.train_and_plot_model(model_name, lr_shape, scale_factor, lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image, epochs, batch_size)
    
    # initialise models
    gan_generator = GAN_generator(lr_shape, scale_factor)
    gan_discriminator = GAN_discriminator(hr_shape)
    gan_discriminator.compile(loss="binary_crossentropy", optimizer="adam")
    gan_discriminator.trainable = False
    srgan_model = SRGAN_block(gan_generator, gan_discriminator, lr_shape,  hr_shape)
    srgan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="Adam")

    # Train the non-VGG GAN
    gan_generator_no_vgg, gan_discriminator_no_vgg, d_loss_history, g_loss_history, psnr_history, d_val_loss_history, g_val_loss_history, val_psnr_history = train_srgan(gan_generator, gan_discriminator, srgan_model, lr_images, hr_images, epochs=epochs, batch_size=batch_size)

    # Plot the generated image
    gen_hr_gan, real_hr_gan, gen_lr_gan = ist.plot_generated_image(gan_generator_no_vgg, 90, lr_images, hr_images)

    # Plot the training and validation diagrams
    plot_metrics(d_loss_history, g_loss_history, psnr_history, d_val_loss_history, g_val_loss_history, val_psnr_history)

    # Calculate PSNR and SSIM
    psnr_value = psnr(real_hr_gan, gen_hr_gan, data_range=real_hr_gan.max() - real_hr_gan.min())
    ssim_value = ssim(real_hr_gan, gen_hr_gan, multichannel=True, data_range=real_hr_gan.max() - real_hr_gan.min(), win_size=3)
    print(f"Non-VGG Average PSNR: {psnr_value}")
    print(f"Non-VGG Average SSIM: {ssim_value}")

    # Calculate PSNR and SSIM for low-resolution image
    psnr_value = psnr(real_hr_gan, gen_lr_gan, data_range=real_hr_gan.max() - real_hr_gan.min())
    ssim_value = ssim(real_hr_gan, gen_lr_gan, multichannel=True, data_range=real_hr_gan.max() - real_hr_gan.min(), win_size=3)
    print(f"Non-VGG Average PSNR for Low-Res Image: {psnr_value}")
    print(f"Non-VGG Average SSIM for Low-Res Image: {ssim_value}")

    # Define the VGG-based GAN
    vgg_network = vgg19_block()
    srgan_model_vgg = SRGAN_block_vgg(gan_generator, gan_discriminator, vgg_network, lr_shape)
    srgan_model_vgg.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="Adam")
    for layer in vgg_network.layers:
        layer.trainable = False
    # Train the VGG-based GAN
    gan_generator_vgg, gan_discriminator_vgg, epoch_g_losses_vgg, epoch_d_losses_vgg, epoch_psnr_values_vgg, val_g_losses_vgg, val_d_losses_vgg, val_psnr_values_vgg = train_srgan_vgg(gan_generator, gan_discriminator, srgan_model_vgg, vgg_network, lr_images, hr_images, epochs=epochs, batch_size=batch_size)

    # Plot the training and validation diagrams
    plot_metrics_vgg(epoch_g_losses_vgg, epoch_d_losses_vgg, epoch_psnr_values_vgg, val_g_losses_vgg, val_d_losses_vgg, val_psnr_values_vgg)

    # Plot the generated image
    gen_hr_ganvgg, real_hr_ganvgg, gen_lr_ganvgg = ist.plot_generated_image(gan_generator_vgg, 90, lr_images, hr_images)

    # Calculate PSNR and SSIM
    psnr_value = psnr(real_hr_ganvgg, gen_hr_ganvgg, data_range=real_hr_ganvgg.max() - real_hr_ganvgg.min())
    ssim_value = ssim(real_hr_ganvgg, gen_hr_ganvgg, multichannel=True, data_range=real_hr_ganvgg.max() - real_hr_ganvgg.min(), win_size=3)
    print(f"Average PSNR: {psnr_value}")
    print(f"Average SSIM: {ssim_value}")

    # Calculate PSNR and SSIM for low-resolution image
    psnr_value = psnr(real_hr_ganvgg, gen_lr_ganvgg, data_range=real_hr_ganvgg.max() - real_hr_ganvgg.min())
    ssim_value = ssim(real_hr_ganvgg, gen_lr_ganvgg, multichannel=True, data_range=real_hr_ganvgg.max() - real_hr_ganvgg.min(), win_size=3)
    print(f"Average PSNR for Low-Res Image: {psnr_value}")
    print(f"Average SSIM for Low-Res Image: {ssim_value}")