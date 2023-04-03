#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:45:48 2023

"""

'''
This script provides a systematic approach to fine-tuning the architecture of Generative Adversarial Networks (GANs) for image super-resolution tasks. 
It includes functions for adjusting model parameters such as layer depths, attention mechanisms, skip connections, and feature extraction algorithms for content loss calculations. 
'''
import sys
sys.path.append('../gan_based_model')
sys.path.append('../initial_model_selection')
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, UpSampling2D, Multiply, MaxPool2D, GlobalAveragePooling2D, Reshape, add
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import gan_construction_training as gct
import initial_selection_training as ist
#%%
def GAN_generator_no_attention(lr_shape, scale_factor):
    """
    Create the GAN generator model with the specified low-resolution shape and scale factor with.
    This generator model does not include the attention mechanism but has skip connections.

    Args:
        lr_shape (tuple): Shape of the low-resolution input image.
        scale_factor (int): Upsampling scale factor.

    Returns:
        generator (Model): The GAN generator model.
    """
    input_layer = Input(shape=lr_shape)
    # SRCNN
    upsampled = UpSampling2D(size=scale_factor, interpolation='bilinear')(input_layer)
    x = Conv2D(64, kernel_size=9, activation='relu', padding='same')(upsampled)
    x = Conv2D(32, kernel_size=1, activation='relu', padding='same')(x)
    x = Conv2D(3, kernel_size=5, padding='same')(x)
    srcnn_output = add([upsampled, x])


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

    decoded_image = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l15)

    decoded_output = add([srcnn_output, decoded_image, upsampled])

    return Model(input_layer, decoded_output)

#%%
def GAN_generator_original(lr_shape, scale_factor):
    """
    Create the GAN generator model with the specified low-resolution shape and scale factor with.
    This generator model does not include the attention mechanism and skip connections.

    Args:
        lr_shape (tuple): Shape of the low-resolution input image.
        scale_factor (int): Upsampling scale factor.

    Returns:
        generator (Model): The GAN generator model.
    """
    input_layer = Input(shape=lr_shape)
    # SRCNN
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

    decoded_image = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l15)

    return Model(input_layer, decoded_image)

#%%
def GAN_generator_cnn(lr_shape, scale_factor):
    """
    Create a GAN generator model with additional CNN layers for hyper-parameter tuning.
    The model tuned the CNN architecture's depth

    Args:
        lr_shape (tuple): Shape of the low-resolution input image.
        scale_factor (int): Upsampling scale factor.

    Returns:
        generator (Model): The GAN generator model with additional CNN layers.
    """
    input_layer = Input(shape=lr_shape)
    
    # SRCNN
    upsampled = UpSampling2D(size=scale_factor, interpolation='bilinear')(input_layer)
    x = Conv2D(64, kernel_size=9, activation='relu', padding='same')(upsampled)
    
    # Additional CNN Layer
    x = Conv2D(64, kernel_size=9, activation='relu', padding='same')(x)
    x = Conv2D(32, kernel_size=1, activation='relu', padding='same')(x)
    
    # Additional CNN Layer
    x = Conv2D(32, kernel_size=1, activation='relu', padding='same')(x)
    x = Conv2D(3, kernel_size=5, padding='same')(x)
    srcnn_output = add([upsampled, x])

    # Self-Attention Mechanism
    channels = 3
    attention = GlobalAveragePooling2D()(srcnn_output)
    attention = Reshape((1, 1, channels))(attention)
    attention = Dense(channels // 2, activation='selu', use_bias=False, kernel_initializer='he_uniform')(attention)
    attention = Dense(channels, activation='selu', use_bias=False, kernel_initializer='he_uniform')(attention)
    attention = Activation('sigmoid')(attention)
    srcnn_output = Multiply()([srcnn_output, attention])

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
    
    decoded_image = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(l15)
    
    decoded_output = add([srcnn_output, decoded_image, upsampled])
    
    channels = 3
    attention = GlobalAveragePooling2D()(srcnn_output)
    attention = Reshape((1, 1, channels))(attention)
    attention = Dense(channels // 2, activation='selu', use_bias=False, kernel_initializer='he_uniform')(attention)
    attention = Dense(channels, activation='selu', use_bias=False, kernel_initializer='he_uniform')(attention)
    attention = Activation('sigmoid')(attention)
    
    output = Multiply()([decoded_output, attention])
    
    return Model(input_layer, output)

#%%
def vgg16_block():
    """
    Creates a VGG16 model with pre-trained ImageNet weights and includes all layers up to block4_conv4.

    Returns:
        Model: The VGG19 model.
    """
    model = VGG16(weights="imagenet", include_top=False, input_shape=(300, 300, 3))
    return Model(inputs=model.inputs, outputs=model.get_layer('block4_conv3').output)

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
def tuning_and_evaluate_gan(lr_images, hr_images, lr_val_image, hr_val_image, epochs, batch_size, gan_generator, gan_discriminator, srgan_model, vgg_network):
    """
    Tunes the hyperparameters of the SRGAN model, trains the model, and evaluates its performance using PSNR and SSIM metrics.
    
    Args:
        lr_images (numpy array): The low-resolution training images.
        hr_images (numpy array): The high-resolution training images.
        lr_val_image (numpy array): The low-resolution validation images.
        hr_val_image (numpy array): The high-resolution validation images.
        epochs (int): The number of epochs for training the model.
        batch_size (int): The size of the mini-batches for training the model.
        gan_generator (Model): The generator model of the SRGAN.
        gan_discriminator (Model): The discriminator model of the SRGAN.
        srgan_model (Model): The combined SRGAN model.
        vgg_network (Model): The VGG19 model for extracting features and calculating loss.
    
    Returns:
        None. Prints the PSNR and SSIM metrics for the generated high-resolution images and plots the training and validation losses.
    """

    gan_discriminator.compile(loss="binary_crossentropy", optimizer="adam")
    gan_discriminator.trainable = False
    srgan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="Adam")

    # Train the VGG-based GAN
    gan_generator_vgg, gan_discriminator_vgg, epoch_g_losses_vgg, epoch_d_losses_vgg, epoch_psnr_values_vgg, val_g_losses_vgg, val_d_losses_vgg, val_psnr_values_vgg = gct.train_srgan_vgg(gan_generator, gan_discriminator, srgan_model, vgg_network, lr_images, hr_images, epochs=epochs, batch_size=batch_size)

    # Plot the training and validation diagrams
    gct.plot_metrics_vgg(epoch_g_losses_vgg, epoch_d_losses_vgg, epoch_psnr_values_vgg, val_g_losses_vgg, val_d_losses_vgg, val_psnr_values_vgg)

    # Plot the generated image
    gen_hr_ganvgg, real_hr_ganvgg, gen_lr_ganvgg = ist.plot_generated_image(gan_generator_vgg, 87, lr_val_image, hr_val_image)

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
    
#%%
def train_and_evaluate_gan_models(lr_images, hr_images, lr_val_image, hr_val_image):
    """
    This function trains and evaluates different GAN models with different generators and VGG networks.
    :param lr_images: Low-resolution images
    :param hr_images: High-resolution images
    """
    lr_shape = lr_images[1].shape
    hr_shape = hr_images[1].shape
    
    # GAN generator with no attention
    generator_no_attention = GAN_generator_no_attention(lr_shape, 4)
    gan_discriminator = gct.GAN_discriminator(hr_shape)
    vgg_19_network = gct.vgg19_block()
    srgan_no_attention = SRGAN_block_vgg(generator_no_attention, gan_discriminator, vgg_19_network, lr_shape)
    tuning_and_evaluate_gan(lr_images, hr_images, lr_val_image, hr_val_image, 20, 1, generator_no_attention, gan_discriminator, srgan_no_attention, vgg_19_network)

    # GAN generator with original architecture
    generator_original = GAN_generator_original(lr_shape, 4)
    srgan_original = SRGAN_block_vgg(generator_original, gan_discriminator, vgg_19_network, lr_shape)
    tuning_and_evaluate_gan(lr_images, hr_images, lr_val_image, hr_val_image, 20, 1, generator_original, gan_discriminator, srgan_original, vgg_19_network)

    # GAN generator with CNN architecture
    generator_cnn = GAN_generator_cnn(lr_shape, 4)
    srgan_cnn = SRGAN_block_vgg(generator_cnn, gan_discriminator, vgg_19_network, lr_shape)
    tuning_and_evaluate_gan(lr_images, hr_images, lr_val_image, hr_val_image, 20, 1, generator_cnn, gan_discriminator, srgan_cnn, vgg_19_network)

    # GAN generator and VGG16 network
    gan_generator = gct.GAN_generator(lr_shape, 4)
    vgg_16_network = vgg16_block()
    srgan_vgg = SRGAN_block_vgg(gan_generator, gan_discriminator, vgg_16_network, lr_shape)
    tuning_and_evaluate_gan(lr_images, hr_images, lr_val_image, hr_val_image, 20, 1, gan_generator, gan_discriminator, srgan_vgg, vgg_16_network)