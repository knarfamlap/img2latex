import tensorflow as tf

from components.positional import  add_timing_signal_nd

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(padding='same'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(padding='same'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='same'),
            tf.keras.layers.Conv2D(512, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Lambda(add_timing_signal_nd),
            tf.keras.layers.Dense(embedding_dim),
        ])
    def call(self, x):
        x = self.encoder(x)
        x = tf.nn.relu(x)

        return x
       