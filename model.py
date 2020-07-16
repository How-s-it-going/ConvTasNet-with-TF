import tensorflow as tf


class BLSTM(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(BLSTM, self).__init__()
        self.lstm = tf.keras.layers.LSTM(dim, go_backwards=True)
        self.linear = tf.keras.layers.Dense(2 * dim * dim)

    def call(self, inputs, **kwargs):
        x = self.lstm(inputs)
        x = self.linear(x)

        return x
