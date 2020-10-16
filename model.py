import tensorflow as tf


class BLSTM(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(BLSTM, self).__init__()
        self.lstm = tf.keras.layers.LSTM(dim, go_backwards=True)
        self.linear = tf.keras.layers.Dense(2 * dim * dim)

    def call(self, inputs, **kwargs):
        inputs = tf.transpose(inputs, perm=[2, 0, 1])
        x = self.lstm(inputs)[0]
        x = self.linear(x)
        x = tf.transpose(x, perm=[1, 2, 0])

        return x


class GLU(tf.keras.layers.Layer):
    def __init__(self):
        super(GLU, self).__init__()
        self.h1 = None


class Demucs(tf.keras.Model):
    def __init__(self,
                 sources=4,
                 channels=64,
                 depth=6,
                 rewrite=True,
                 glu=True,
                 upsample=False,
                 rescale=0.1,
                 kernel_size=8,
                 stride=4,
                 growth=2.,
                 lstm_layers=2,
                 context=3):
        super(Demucs, self).__init__()
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.upsample = upsample
        self.channels = channels

        self.encoder = []
        self.decoder = []

        self.final = None
        if upsample:
            self.final = tf.keras.layers.Conv1D(sources * 2, 1)
            stride = 1

        if glu:
            print()
