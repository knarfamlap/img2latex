import tensorflow as tf

from components.positional import  add_timing_signal_nd

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()

        self.conv2d_1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(50, 120, 3))
        self.maxpool_1 = tf.keras.layers.MaxPool2D(padding='same')

        self.conv2d_2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.maxpool_2 = tf.keras.layers.MaxPool2D(padding='same')

        self.conv2d_3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')

        self.conv2d_4 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')

        self.conv2d_5 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')

        self.conv2d_6 = tf.keras.layers.Conv2D(512,  2, strides=(2, 4), activation='relu')

        self.conv2d_7 = tf.keras.layers.Conv2D(512,  3, activation='relu')

        self.signal = tf.keras.layers.Lambda(add_timing_signal_nd)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.conv2d_1(x)
        x = self.maxpool_1(x)

        x = self.conv2d_2(x)
        x = self.maxpool_2(x)

        x = self.conv2d_3(x)

        x = self.conv2d_4(x)

        x = self.conv2d_5(x)

        x = self.conv2d_6(x)

        x = self.conv2d_7(x)

        x = self.signal(x)

        x = self.flatten(x)

        x = self.dense(x)


        return x
       