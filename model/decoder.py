import tensorflow as tf
import tensorflow_addons as tfa

from .attention import BahdanauAttention


class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, id_end):
        super(Decoder, self). __init__()
        self.units = units
        # comes from Vocab
        self._id_end = id_end

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTMCell(self.units)

        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

        self.start_token = tf.Variable(initial_value=embedding_initializer(embedding_dim, dtype=tf.float32), dtype=tf.float32, shape=[embedding_dim],)

        self.decoder = tfa.seq2seq.BeamSearchDecoder(
            self.lstm, beam_width=3, output_layer=self.fc1, )

    def call(self, x, features, hidden):
        # defining attention as seperate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passsing concatenated vector to LSTM
        output, state = self.decoder(x, start_tokens=self.start_token)

        # shape == (batch_size, max_length, hidden_size)
        # x = self.fc1(output)

        # x shape == (batch_size, max_length, hidden_size)
        x = tf.reshape(output, (-1, x.shape[2]))

        # output shape == (batch_size, * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def embedding_initializer(shape, dtype):
    E = tf.random.uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
    E = tf.nn.l2_normalize(E, -1)
    return E

