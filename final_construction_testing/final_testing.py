#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 05:05:43 2023

"""
import sys
sys.path.append('../gan_based_model')
sys.path.append('../initial_model_selection')
sys.path.append('../initial_model_construction')
import gan_construction_training as gct
import initial_selection_training as ist
import initial_model_construction as imc
import numpy as np
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
def evaluate_generator(generator, hr_val_images, lr_val_images, example_idx=0, use_bicubic=False):
    """
    This function evaluates the performance of a super-resolution generator model on given validation images.
    It calculates PSNR, SSIM, and MSE between the high-resolution (HR) images and the generated HR images.
    The function also plots HR, generated HR, and low-resolution (LR) images for visual comparison.

    :param generator: The super-resolution model to evaluate
    :param hr_val_images: List of high-resolution validation images
    :param lr_val_images: List of low-resolution validation images
    :param example_idx: Index of the image to be plotted for visual comparison (default: 0)
    :param use_bicubic: Boolean to indicate whether to upsample LR images using bicubic interpolation (default: False)
    """
    hr_val = np.array(hr_val_images)
    lr_val = np.array(lr_val_images)

    # Upsample LR images using bicubic interpolation if use_bicubic is True
    if use_bicubic:
        bicubic_val = []
        for img in lr_val:
            img = img_to_array(img)
            img = img.astype('uint8')
            img_resized = cv2.resize(img, (hr_val.shape[2], hr_val.shape[1]), interpolation=cv2.INTER_CUBIC)
            bicubic_val.append(img_resized)
        bicubic_val = np.array(bicubic_val)
        generated_hr_val = generator.predict_on_batch(bicubic_val)
    else:
        generated_hr_val = generator.predict_on_batch(lr_val)

    # Get HR, generated HR, and LR images using the specified example index
    hr_img = hr_val[example_idx]
    gen_hr_img = generated_hr_val[example_idx]
    lr_img = lr_val[example_idx]

    # Calculate PSNR, SSIM, and MSE between HR and generated HR images
    psnr_val = psnr(hr_img, gen_hr_img)
    ssim_val = structural_similarity(hr_img, gen_hr_img, multichannel=True)
    mse_val = mean_squared_error(hr_img, gen_hr_img)

    # Plot HR, generated HR, and LR images
    plt.figure(figsize=(50, 20))

    plt.subplot(131)
    plt.title('HR')
    plt.imshow(hr_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Generated HR')
    plt.imshow(gen_hr_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('LR')
    plt.imshow(lr_img)
    plt.axis('off')

    plt.suptitle(f'PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.2f}, MSE: {mse_val:.2f}')

    plt.show()

#%%
def compile_and_train_final(model, lr_images, hr_images, epochs, batch_size):
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
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mse', 'accuracy', ist.psnr_metric, ist.ssim_metric])
    history = model.fit(x=lr_images, y=hr_images, batch_size=batch_size, epochs=epochs)
    return history

#%%
def train_and_evaluate_srcnn(scale_factor, lr_crop_image, hr_crop_image, lr_val_image, hr_val_image, example_idx=91, use_bicubic=False):
    """
        Train and evaluate the SRCNN model on given dataset.
        
        Args:
            scale_factor (int): The scale factor for super-resolution.
            lr_crop_image (numpy.ndarray): Low-resolution cropped images.
            hr_crop_image (numpy.ndarray): High-resolution cropped images.
            lr_val_image (numpy.ndarray): Low-resolution validation images.
            hr_val_image (numpy.ndarray): High-resolution validation images.
            example_idx (int, optional): Index of the example image to visualize. Defaults to 91.
            use_bicubic (bool, optional): Whether to use bicubic interpolation for comparison. Defaults to False.
        
        Returns:
            None
    """
    lr_shape = lr_crop_image[1].shape
    srcnn_model = imc.srcnn(lr_shape, scale_factor)
    batch_size = 1
    epochs = 20
    srcnn_history = compile_and_train_final(srcnn_model, lr_crop_image, hr_crop_image, epochs, batch_size)
    evaluate_generator(srcnn_model, hr_val_image, lr_val_image, example_idx, use_bicubic)

#%%
def train_and_evaluate_autoencoder(lr_crop_image, hr_crop_image, lr_val_image, hr_val_image, scale_factor, example_idx=91, use_bicubic=False):
    """
    Train and evaluate the autoencoder model on given dataset.
    
        Args:
            lr_crop_image (numpy.ndarray): Low-resolution cropped images.
            hr_crop_image (numpy.ndarray): High-resolution cropped images.
            lr_val_image (numpy.ndarray): Low-resolution validation images.
            hr_val_image (numpy.ndarray): High-resolution validation images.
            scale_factor (int): The scale factor for super-resolution.
            example_idx (int, optional): Index of the example image to visualize. Defaults to 91.
            use_bicubic (bool, optional): Whether to use bicubic interpolation for comparison. Defaults to False.
        
        Returns:
            None
    """
    lr_shape = lr_crop_image[1].shape
    lr_images_resized = imc.resize_images(lr_crop_image, scale_factor)
    autoencoder_model = imc.autoencoder()
    batch_size = 1
    epochs = 20
    autoencoder_history = compile_and_train_final(autoencoder_model, lr_images_resized, hr_crop_image, epochs, batch_size)
    lr_val_resized = imc.resize_images(lr_val_image, scale_factor)
    evaluate_generator(autoencoder_model, hr_val_image, lr_val_resized, example_idx, use_bicubic)


#%%
def create_gan(lr_shape, hr_shape, scale_factor):
    """
        Create GAN generator, discriminator, SRGAN model, and VGG19 network for feature extraction.
        
        Args:
            lr_shape (tuple): Shape of the low-resolution images.
            hr_shape (tuple): Shape of the high-resolution images.
            scale_factor (int): The scale factor for super-resolution.
        
        Returns:
            Tuple: The GAN generator, discriminator, SRGAN model, and VGG19 network.
    """
    gan_generator = gct.GAN_generator(lr_shape, scale_factor)
    gan_discriminator = gct.GAN_discriminator(hr_shape)
    vgg_network = gct.vgg19_block()

    # Compile the discriminator
    discriminator_optimizer = Adam(lr=1e-4, beta_1=0.9)
    gan_discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)

    # Freeze discriminator and VGG19 layers for SRGAN training
    gan_discriminator.trainable = False
    for layer in vgg_network.layers:
        layer.trainable = False

    # Compile the SRGAN model
    srgan_model = gct.SRGAN_block_vgg(gan_generator, gan_discriminator, vgg_network, lr_shape)
    srgan_optimizer = Adam(lr=1e-4, beta_1=0.9)
    srgan_model.compile(loss=['binary_crossentropy', 'mse'], optimizer=srgan_optimizer, loss_weights=[1e-3, 1])

    return gan_generator, gan_discriminator, srgan_model, vgg_network


#%%
def train_srgan_vgg_final(gan_generator, gan_discriminator, srgan_model, vgg_network, lr_images, hr_images, epochs=30, batch_size=4):
    """
    Train the SRGAN model on the given dataset using the VGG19 loss.

    Args:
        gan_generator (Model): Generator model.
        gan_discriminator (Model): Discriminator model.
        srgan_model (Model): SRGAN model.
        vgg_network (Model): VGG19 model for feature extraction.
        lr_images (numpy.ndarray): Low-resolution images.
        hr_images (numpy.ndarray): High-resolution images.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        Tuple: The trained generator, discriminator, and their loss and PSNR histories.
    """

    # Create batches of low-resolution and high-resolution images for training
    train_lr_batches = []
    train_hr_batches = []

    for it in range(int(len(hr_images) / batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        train_hr_batches.append(hr_images[start_idx:end_idx])
        train_lr_batches.append(lr_images[start_idx:end_idx])

    # Initialize lists to store losses and PSNR values
    epoch_g_losses = []
    epoch_d_losses = []
    epoch_psnr_values = []
    
    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
    
        g_losses = []
        d_losses = []
    
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
    
        # Compute mean losses and PSNR values
        g_loss = np.mean(g_losses)
        d_loss = np.mean(d_losses)
        psnr_value = np.mean([psnr(hr_img, fake_img, data_range=hr_img.max() - hr_img.min()) for hr_img, fake_img in zip(hr_imgs, fake_imgs)])
    
        # Append losses and PSNR values to history
        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss)
        epoch_psnr_values.append(psnr_value)
    
        # Print current losses and PSNR values
        print(f"Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}, PSNR: {psnr_value:.4f}")
    
        # Return the trained generator
    return gan_generator


#%%
def train_and_evaluate_srgan(scale_factor, lr_crop_image, hr_crop_image, lr_val_image, hr_val_image, example_idx=91, use_bicubic=False):
    """
    Train and evaluate the SRGAN model on given dataset.

    Args:
        scale_factor (int): The scale factor for super-resolution.
        lr_crop_image (numpy.ndarray): Low-resolution cropped images.
        hr_crop_image (numpy.ndarray): High-resolution cropped images.
        lr_val_image (numpy.ndarray): Low-resolution validation images.
        hr_val_image (numpy.ndarray): High-resolution validation images.
        example_idx (int, optional): Index of the example image to visualize. Defaults to 91.
        use_bicubic (bool, optional): Whether to use bicubic interpolation for comparison. Defaults to False.

    Returns:
        None
    """
    # Get the shape of the low-resolution and high-resolution images
    lr_shape = lr_crop_image[1].shape
    hr_shape = hr_crop_image[1].shape
    
    # Create GAN generator, discriminator, SRGAN model, and VGG19 network
    gan_generator, gan_discriminator, srgan_model, vgg_network = create_gan(lr_shape, hr_shape, scale_factor)
    
    # Train the SRGAN model
    gan_generator_vgg = train_srgan_vgg_final(gan_generator, gan_discriminator, srgan_model, vgg_network, lr_crop_image, hr_crop_image, epochs=20, batch_size=1)
    
    # Evaluate the trained SRGAN model
    evaluate_generator(gan_generator_vgg, hr_val_image, lr_val_image, example_idx, use_bicubic)


#%%
def train_and_evaluate_all_models(hr_crop_image, hr_val_image, hr_crop_image_un, hr_val_image_un):
    """
    Train and evaluate all models (SRCNN, autoencoder, SRGAN) for different scale factors.

    Args:
        hr_crop_image (numpy.ndarray): High-resolution cropped images.
        hr_val_image (numpy.ndarray): High-resolution validation images.
        hr_crop_image_un (numpy.ndarray): High-resolution cropped images (unseen dataset).
        hr_val_image_un (numpy.ndarray): High-resolution validation images (unseen dataset).

    Returns:
        None
    """
    # Define the scale factors and models for evaluation
    scale_factors = [2, 3, 4]
    models = ['SRCNN', 'autoencoder', 'SRGAN']

    # Iterate through each model and scale factor combination
    for model in models:
        for scale_factor in scale_factors:
            print(f"Training and evaluating {model} for scale factor {scale_factor}")

            # Get the corresponding low-resolution images for the current scale factor
            lr_crop_image = globals()[f'lr_x{scale_factor}_crop_image']
            lr_val_image = globals()[f'lr_x{scale_factor}_val_image']
            lr_crop_image_un = globals()[f'lr_x{scale_factor}_crop_image_un']
            lr_val_image_un = globals()[f'lr_x{scale_factor}_val_image_un']

            # Train and evaluate the selected model with the current scale factor
            if model == 'SRCNN':
                train_and_evaluate_srcnn(scale_factor, lr_crop_image, hr_crop_image, lr_val_image, hr_val_image)
                train_and_evaluate_srcnn(scale_factor, lr_crop_image_un, hr_crop_image_un, lr_val_image_un, hr_val_image_un)
            elif model == 'autoencoder':
                train_and_evaluate_autoencoder(lr_crop_image, hr_crop_image, lr_val_image, hr_val_image, scale_factor)
                train_and_evaluate_autoencoder(lr_crop_image_un, hr_crop_image_un, lr_val_image_un, hr_val_image_un, scale_factor)
            elif model == 'SRGAN':
                train_and_evaluate_srgan(scale_factor, lr_crop_image, hr_crop_image, lr_val_image, hr_val_image)
                train_and_evaluate_srgan(scale_factor, lr_crop_image_un, hr_crop_image_un, lr_val_image_un, hr_val_image_un)
