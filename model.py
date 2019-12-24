import tensorflow as tf
import numpy as np
import librosa
import hyperparamator as hp


class ConvTasNet:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()


class Encoder:
    def __init__(self):
        self.conv1d = tf.keras.layers.Conv1D(hp.N, kernel_size=hp.L, strides=hp.L // 2, use_bias=False)

    def forward(self, mixture):
        mixture = tf.expand_dims(mixture, axis=1)  # [M, 1, T]
        mixture_w = tf.nn.relu(self.conv1d(mixture))  # [M, N, T]
        return mixture_w


class Decoder:
    def __init__(self):
        self.basis_signals = tf.keras.layers.Dense(hp.L, use_bias=False)

    def forward(self, mixture_w, est_mask):
        source_w = tf.expand_dims(mixture_w, axis=1) * est_mask  # [M, C, N, K]
        source_w = tf.transpose(source_w, perm=[0, 1, 3, 2])  # [M, C, K, N]

        est_source = self.basis_signals(source_w)  # [M, C, K, L]
        est_source = tf.signal.overlap_and_add(est_source, hp.L // 2)  # M * C * T
        return est_source


class TemporalConvNet:
    def __init__(self):
        layer_norm = ChannelwiseLayerNorm(hp.N)  # [M, N, K]
        bottleneck_conv1d = tf.keras.layers.Conv1D(hp.B, 1, use_bias=False)  # [M, B, K]
        repeats = []
        for r in range(hp.R):
            blocks = []
            for x in range(hp.X):
                dilation = 2**x
                padding = (hp.P - 1) * dilation if hp.causal else (hp.P - 1) * dilation // 2


class TemporalBlock:
    def __init__(self, filters, kernel_size):
        conv1d = tf.keras.layers.Conv1D(filters, 1, use_bias=False)
        prelu = tf.keras.layers.PReLU()


class ChannelwiseLayerNorm:
    def __init__(self, channel_size):
        self.gamma = tf.Variable(1, trainable=True, shape=[1, channel_size, 1], name='gamma')
        self.beta = tf.Variable(0, trainable=True, shape=[1, channel_size, 1], name='beta')

    def forward(self, y):
        mean, var = tf.nn.moments(y, axes=1, keep_dims=True)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / tf.pow(var + hp.epsilon, 0.5) + self.beta
        return cLN_y
