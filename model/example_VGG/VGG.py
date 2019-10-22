import tensorflow as tf
from Config import*
from data_generator import*
import os
import random
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import plot_model


def model_(input_shape, classes):

    model = VGG16(include_top=False, input_shape=input_shape)
    x = Flatten()(model.output)
    x = Dense(1024, activation='relu', name='ft1')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', name='ft2')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    
    custom_model = Model(inputs=model.input, outputs=x)
    custom_model.summary()

    custom_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']) 
    
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    custom_model.save('VGG.h5')

if  __name__ == "__main__":
    model_((250, 250, 3), 3)
