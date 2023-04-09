#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 23:13:04 2023

@author: ericwei
"""

import sys
sys.path.append('../gan_based_model')
sys.path.append('../initial_model_selection')
import gan_construction_training as gct
import tensorflow as tf
from kerastuner import RandomSearch
from tqdm import tqdm
import numpy as np
import initial_selection_training as ist
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
#%%
def gan_tuning(lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image, max_trials=10, search_epochs=20, train_epochs=20, batch_size=8):
    """
        Tune and train a GAN model for image super-resolution with the given parameters.
        This function performs hyperparameter tuning using Keras Tuner and trains the GAN model for the specified number of epochs.
        
        Args:
            lr_x4_crop_image (np.array): Low-resolution training images.
            hr_crop_image (np.array): High-resolution training images.
            lr_x4_val_image (np.array): Low-resolution validation images.
            hr_val_image (np.array): High-resolution validation images.
            max_trials (int, optional): Maximum number of hyperparameter search trials. Defaults to 10.
            search_epochs (int, optional): Number of epochs to search for hyperparameters. Defaults to 20.
            train_epochs (int, optional): Number of epochs to train the GAN model. Defaults to 20.
            batch_size (int, optional): Batch size for training. Defaults to 8.
        
        Returns:
            None
    """
    def build_gan(hp):
        lr_shape = (75, 75, 3)
        hr_shape = (300, 300, 3)
        scale_factor = 4

        # Define the generator
        generator = gct.GAN_generator(lr_shape, scale_factor)

        # Define the discriminator
        discriminator = gct.GAN_discriminator(hr_shape)

        # Set up hyperparameters to tune
        gen_lr = hp.Float('gen_learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        gen_beta_1 = hp.Float('gen_beta_1', min_value=0.0, max_value=1.0, step=0.1)

        disc_lr = hp.Float('disc_learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        disc_beta_1 = hp.Float('disc_beta_1', min_value=0.0, max_value=1.0, step=0.1)

        gan_lr = hp.Float('gan_learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        gan_beta_1 = hp.Float('gan_beta_1', min_value=0.0, max_value=1.0, step=0.1)

        # Compile the generator
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=gen_beta_1)
        generator.compile(loss='mse', optimizer=gen_optimizer)

        # Compile the discriminator
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=disc_beta_1)
        discriminator.compile(loss='binary_crossentropy', optimizer=disc_optimizer)

        # Define the GAN model (stacking the generator and discriminator)
        vgg19 = gct.vgg19_block()
        gan = gct.SRGAN_block_vgg(generator, discriminator, vgg19, lr_shape)

        # Compile the GAN model
        gan_optimizer = tf.keras.optimizers.Adam(learning_rate=gan_lr, beta_1=gan_beta_1)
        gan.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[1, 1e-3], optimizer=gan_optimizer)

        return gan
    
    
    def train_gan(tuner, x, y, epochs, batch_size, validation_data):
        # Get the best GAN model from the tuner
        best_gan = tuner.get_best_models()[0]
    
        # Train the GAN model for the specified number of epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
    
            # Initialize lists to store generator and discriminator losses
            g_losses = []
            d_losses = []
    
            # Iterate over the dataset in batches
            for i in tqdm(range(0, x.shape[0], batch_size)):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
    
                # Train the GAN model on the current batch
                g_loss = best_gan.train_on_batch(x_batch, [y_batch, np.ones((batch_size, 1))])
                d_loss = best_gan.test_on_batch(x_batch, [y_batch, np.ones((batch_size, 1))])
    
                # Append generator and discriminator losses
                g_losses.append(g_loss[0])
                d_losses.append(d_loss[0])
    
            # Calculate and print average generator and discriminator losses for the current epoch
            avg_g_loss = sum(g_losses) / len(g_losses)
            avg_d_loss = sum(d_losses) / len(d_losses)
            print(f"Generator Loss: {avg_g_loss:.4f}, Discriminator Loss: {avg_d_loss:.4f}")
    
            # Evaluate the GAN model on validation data every 5 epochs
            if (epoch + 1) % 5 == 0:
                # Initialize lists to store validation generator and discriminator losses
                val_g_loss = []
                val_d_loss = []
    
                # Iterate over the validation dataset in batches
                for i in tqdm(range(0, validation_data[0].shape[0], batch_size)):
                    x_val_batch = validation_data[0][i:i+batch_size]
                    y_val_batch = validation_data[1][i:i+batch_size]
    
                    # Evaluate the GAN model on the current validation batch
                    val_g_loss.append(best_gan.evaluate(x_val_batch, [y_val_batch, np.ones((batch_size, 1))])[0])
                    val_d_loss.append(best_gan.evaluate(x_val_batch, [y_val_batch, np.ones((batch_size, 1))])[1])
    
                # Calculate and print average validation generator and discriminator losses
                avg_val_g_loss = sum(val_g_loss) / len(val_g_loss)
                avg_val_d_loss = sum(val_d_loss) / len(val_d_loss)
                print(f"Validation Loss: Generator Loss: {avg_val_g_loss:.4f}, Discriminator Loss: {avg_val_d_loss:.4f}")

    
    tuner = RandomSearch(
        build_gan,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='gan_tuning'
    )
    tuner.search_space_summary()

    tuner.search(
        x=lr_x4_crop_image,
        y=[hr_crop_image, np.ones((lr_x4_crop_image.shape[0], 1))],
        epochs=search_epochs,
        batch_size=batch_size,
        validation_data=(lr_x4_val_image, [hr_val_image, np.ones((lr_x4_val_image.shape[0], 1))])
    )

    tuner.results_summary()

    train_gan(tuner, lr_x4_crop_image, hr_crop_image, epochs=train_epochs, batch_size=1, validation_data=(lr_x4_val_image, hr_val_image))

#%%
def train_and_evaluate_srgan(batch_size, lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image):
  lr_shape = (75, 75, 3)
  hr_shape = (300, 300, 3)
  scale_factor = 4
  gan_generator = gct.GAN_generator(lr_shape, scale_factor)
  gan_discriminator = gct.GAN_discriminator(hr_shape)
  gan_discriminator.compile(loss="binary_crossentropy", optimizer="adam")
  gan_discriminator.trainable = False
  vgg_network = gct.vgg19_block()
  srgan_model = gct.SRGAN_block_vgg(gan_generator, gan_discriminator, vgg_network, lr_shape)
  srgan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="Adam")
  for layer in vgg_network.layers:
    layer.trainable = False

  trained_gen, trained_disc, epoch_g_losses, epoch_d_losses, epoch_psnr_values, val_g_losses, val_d_losses, val_psnr_values = gct.train_srgan_vgg(gan_generator, gan_discriminator, srgan_model, vgg_network, lr_x4_crop_image, hr_crop_image, val_split=0.1, epochs=20, batch_size=batch_size)
  # Plot the generated image
  gen_hr_gan, real_hr_gan, gen_lr_gan = ist.plot_generated_image(trained_gen, 1, lr_x4_val_image, hr_val_image)

  # Plot the training and validation diagrams
  gct.plot_metrics_vgg(epoch_g_losses, epoch_d_losses, epoch_psnr_values, val_g_losses, val_d_losses, val_psnr_values)


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


#%%
def batch_size_tuning(lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image):
    batch_sizes = [2, 4, 6, 8]
    for batch_size in batch_sizes:
        print(f"Training with batch size: {batch_size}")
        train_and_evaluate_srgan(batch_size, lr_x4_crop_image, hr_crop_image, lr_x4_val_image, hr_val_image)