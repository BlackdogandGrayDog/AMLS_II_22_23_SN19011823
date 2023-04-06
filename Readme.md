# Super Resolution with GAN-based SRCNNencoder

This repository contains the implementation of a GAN-based SRCNNencoder model for image super-resolution. The project aims to develop a model that outperforms existing SRCNN and SRAutoencoder methods in terms of visual perception on the DIV2K dataset.


## Introduction

The field of super-resolution has seen significant advancements in recent years, with numerous methods being proposed to enhance the quality of low-resolution images. In this project, we present a GAN-based SRCNNencoder model that leverages the strengths of both SRCNN and SRAutoencoder approaches to achieve superior results in terms of visual perception. Our model is designed to capture intricate high-level structures and patterns while preserving low-level textures, ultimately resulting in more visually appealing super-resolution images.


## Contents
This repository contains various scripts and modules that cover different aspects of constructing, training, and evaluating deep learning models for image super-resolution tasks.

crop_image_preprocessing: This script provides functions for cropping and reading high-resolution images and their corresponding low-resolution images at different scaling factors (x2, x3, and x4) for use in a super-resolution GAN.

initial_model_construction: This script contains multiple deep learning model architectures for image super-resolution tasks. The main goal of these models is to upscale low-resolution images to high-resolution images while maintaining or improving image quality. This file includes the following architectures:

SRCNN (Super-Resolution Convolutional Neural Network)
SRResNet (Super-Resolution Residual Network)
Autoencoder-based Super-Resolution
Combined SRCNN and Autoencoder
initial_selection_training: This script streamlines the initial model selection, training, and performance comparison process for image super-resolution tasks. It includes functions for training SRCNN, SRResNet, AutoEncoder, compiling and training these models, and visualising generated images and performance metrics.

gan_construction_training: This script provides a comprehensive workflow for constructing, training, and evaluating Generative Adversarial Networks (GANs) for image super-resolution tasks. It includes functions for building the SRGAN architecture with optional VGG19 integration, training the generator and discriminator models, and visualizing the generated high-resolution images along with performance metrics.

architecture_tuning: This script provides a systematic approach to fine-tuning the architecture of Generative Adversarial Networks (GANs) for image super-resolution tasks. It includes functions for adjusting model parameters such as layer depths, attention mechanisms, skip connections, and feature extraction algorithms for content loss calculations.

final_testing: This script includes functions for training the final models on the entire training dataset and evaluating their performance on a separate validation dataset. It compares the performance of our GAN-based SRCNNencoder model with other popular models, SRCNN and SRAutoencoder.

hyperparameter_tuning: This script provides methods for optimizing the hyperparameters of the models, using techniques such as grid search, random search, and Bayesian optimization.

main: The main script that orchestrates the entire process, from data preprocessing to model training, evaluation, and comparison.

## Dependencies

To set up the environment and install the required packages, please create a new conda environment with the following dependencies:

  - python=3.9
  - pip
  - pip:
    - numpy
    - scipy
    - pandas
    - scikit-learn
    - matplotlib
    - tqdm
    - datetime
    - tensorflow
    - keras
    - keras-rl2
    - spyder-kernels
    - OpenCV-python
    - keras-applications
    - keras-preprocessing
    - scikit-image
    - keras-tuner



## Additional Information

Due to limitations, all the metric plotting and SR images generated in this project are attached in the Image folder of the GitHub repository, inside folders with corresponding model names.

## Dataset

The dataset used for training can be download from:
https://www.dropbox.com/sh/23up6jvhwrellrv/AABVBf_XV8F_c7ZZ8I-aL64wa?dl=0

Backup: https://data.vision.ee.ethz.ch/cvl/DIV2K/

For more infomation, please refer to the Readme.md file in the dataset folder.

