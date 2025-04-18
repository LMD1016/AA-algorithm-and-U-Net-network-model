# -*- coding: utf-8 -*-
"""
#Script for U-Net training
Last modified on 1/21/2025
Written by Mingdi Liu
Contact: Mingdi Liu (DD1359406536@163.com)
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def unet(input_shape=(512, 512, 1)):
    inputs = layers.Input(input_shape)

    # Encoder part (downsampling)
    conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    # Bottom part (bottleneck layer)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    # Decoder part (upsampling)
    up6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([up6, conv4], axis=3)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([up7, conv3], axis=3)
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([up8, conv2], axis=3)
    conv8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([up9, conv1], axis=3)
    conv9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)

    # Output layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = models.Model(inputs=inputs, outputs=output)
    return model







