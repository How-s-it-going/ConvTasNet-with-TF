import tensorflow as tf


def rescale_weitghts():
    print()


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
        self.h1 = tf.keras.layers.Conv1D(128, 1)
        self.h1_gates = tf.keras.layers.Conv1D(128, 1)
        self.h1_glu = tf.keras.layers.Multiply()

    def call(self, inputs, **kwargs):
        h1 = self.h1(inputs)


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

        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()

        self.final = None
        if upsample:
            self.final = tf.keras.layers.Conv1D(sources, 1)
            stride = 1

        if glu:
            activation = GLU()
            ch_scale = 2
        else:
            activation = tf.keras.layers.ReLU()
            ch_scale = 1

        in_channels = 1
        for idx in range(depth):
            self.encoder.add(tf.keras.layers.Conv1D(channels, kernel_size, stride))
            self.encoder.add(tf.keras.layers.ReLU())
            if rewrite:
                self.encoder.add(tf.keras.layers.Conv1D(ch_scale * channels, 1))
                self.encoder.add(activation)

            if idx > 0:
                filters = in_channels
            else:
                if upsample:
                    filters = channels
                else:
                    filters = sources

            if rewrite:
                self.decoder.add(tf.keras.layers.Conv1D(ch_scale * channels, context))
                self.decoder.add(activation)

            if upsample:
                self.decoder.add(tf.keras.layers.Conv1D(filters, kernel_size))
            else:
                self.decoder.add(tf.keras.layers.Conv2DTranspose(filters, (1, kernel_size, stride)))

            if idx > 0:
                self.decoder.add(tf.keras.layers.ReLU())

            in_channels = channels
            channels = int(growth * channels)

        if lstm_layers:
            self.lstm = BLSTM(lstm_layers)
        else:
            self.lstm = None

        if rescale:
            print()
