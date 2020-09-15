import numpy as np
import tensorflow as tf


class Decoder(self):
    def __init__(self):
        super().__init__()

    def call(seq):
        img_mean = tf.reduce_mean(seq, axis=1)

        W1_e = tf.keras.layers.Dense(512)(seq)
        W2_h = tf.keras.layers.Dense(512)()