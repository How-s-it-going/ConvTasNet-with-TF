import tensorflow as tf
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
                dilation = 2 ** x
                blocks += [TemporalBlock(hp.B, hp.H, hp.P, stride=1, dilation=dilation)]

            repeats += [tf.keras.Sequential(blocks)]

        temporal_conv_net = tf.keras.Sequential(repeats)
        mask_conv = tf.keras.layers.Conv1D(hp.C * hp.N, 1, use_bias=False)  # [M, C * N, K]

        self.network = tf.keras.Sequential([layer_norm, bottleneck_conv1d, temporal_conv_net, mask_conv])

    def forward(self, mixture_w):
        shape = tf.shape(mixture_w)
        M, N, K = shape[0], shape[1], shape[2]
        score = self.network(mixture_w)  # [M, C * N, K]
        score = tf.reshape(score, shape=[M, hp.C, N, K])  # [M, C, N, K]
        est_mask = tf.keras.layers.ReLU(score)

        return est_mask


class TemporalBlock:
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        conv1d = tf.keras.layers.Conv1D(out_channels, 1, use_bias=False)
        prelu = tf.keras.layers.PReLU()
        norm = chose_norm(hp.norm_type, out_channels)
        dsconv = tf.keras.layers.SeparableConv1D(in_channels, kernel_size, stride,
                                                 padding='causal' if hp.causal else 'same',
                                                 dilation_rate=dilation, use_bias=False,
                                                 activation=tf.keras.layers.PReLU)
        norm2 = chose_norm(hp.norm_type, in_channels)

        self.net = tf.keras.Sequential([conv1d, prelu, norm, dsconv, norm2])

    def forward(self, x):
        residual = x
        out = self.net(x)
        return out + residual


class ChannelwiseLayerNorm:
    def __init__(self, channel_size):
        self.gamma = tf.Variable(1, trainable=True, shape=[1, channel_size, 1], name='gamma')
        self.beta = tf.Variable(0, trainable=True, shape=[1, channel_size, 1], name='beta')

    def forward(self, y):
        mean, var = tf.nn.moments(y, axes=1, keep_dims=True)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / tf.pow(var + hp.epsilon, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm:
    def __init__(self, channel_size):
        self.gamma = tf.Variable(1, trainable=True, shape=[1, channel_size, 1], name='gamma')
        self.beta = tf.Variable(0, trainable=True, shape=[1, channel_size, 1], name='beta')

    def forward(self, y):
        mean = tf.reduce_mean(y, axis=[1, 2], keepdims=True)  # [M, 1, 1]
        var = tf.reduce_mean(tf.pow((y - mean), 2), axis=[1, 2], keepdims=True)  # [M, 1, 1]
        gLN_y = self.gamma * (y - mean) / tf.pow(var + hp.epsilon, 0.5) + self.beta
        return gLN_y


def chose_norm(norm_type, channel_size, is_training=False):
    if norm_type == 'gLN':
        return GlobalLayerNorm(channel_size)
    elif norm_type == 'cLN':
        return ChannelwiseLayerNorm(channel_size)
    else:
        return tf.keras.layers.BatchNormalization(axis=1, trainable=is_training)  # BN.
