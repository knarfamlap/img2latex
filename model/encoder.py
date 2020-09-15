import numpy as np
import tensorflow as tf

from components.positional import  add_timing_signal_nd

class Encoder(object):
    def __init__(self):
        super().__init__()

    def call(self): 
        model = tf.keras.models.Sequential()
        
        model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(padding='same'))

        model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(padding='same'))

        model.add(tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'))
        
        model.add(tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same'))

        model.add(tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='same')) 

        model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))

        model.add(tf.keras.layers.Lambda(add_timing_signal_nd))


        return model