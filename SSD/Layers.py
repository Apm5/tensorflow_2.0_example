import tensorflow as tf
import numpy as np


class FullyConnect(tf.keras.layers.Layer):

    def __init__(self, output_dim, activation=tf.nn.relu, use_bias=True, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        super(FullyConnect, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[-1], self.output_dim))
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer=tf.initializers.RandomUniform,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                          shape=[self.output_dim,],
                                          initializer=tf.initializers.zeros,
                                          trainable=True)
        super(FullyConnect, self).build(input_shape)

    def call(self, inputs):
        if self.use_bias:
            output = tf.add(tf.matmul(inputs, self.kernel), self.bias)
        else:
            output = tf.matmul(inputs, self.kernel)
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, output_dim, kernel=(3, 3), use_bias=True, strides=(1, 1, 1, 1), padding='SAME', **kwargs):
        self.output_dim = output_dim
        self.kernel = kernel
        self.use_bias = use_bias
        self.strides = strides
        self.padding = padding
        super(Conv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((self.kernel[0], self.kernel[1], input_shape[-1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer=tf.initializers.RandomUniform,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=[self.output_dim, ],
                                        initializer=tf.initializers.zeros,
                                        trainable=True)
        super(Conv2D, self).build(input_shape)

    def call(self, inputs):
        if self.use_bias:
            output = tf.add(tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding), self.bias)
        else:
            output = tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding)
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

class DilatedConv2D(tf.keras.layers.Layer):
    def __init__(self, output_dim, rate, kernel=(3, 3), use_bias=True, padding='SAME', **kwargs):
        self.output_dim = output_dim
        self.rate = rate
        self.kernel = kernel
        self.use_bias = use_bias
        self.padding = padding
        super(DilatedConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((self.kernel[0], self.kernel[1], input_shape[-1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer=tf.initializers.RandomUniform,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=[self.output_dim, ],
                                        initializer=tf.initializers.zeros,
                                        trainable=True)
        super(DilatedConv2D, self).build(input_shape)

    def call(self, inputs):
        if self.use_bias:
            output = tf.add(tf.nn.atrous_conv2d(inputs, self.kernel, rate=self.rate, padding=self.padding), self.bias)
        else:
            output = tf.nn.conv2d(inputs, self.kernel, rate=self.rate, padding=self.padding)
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

class L2Normalization(tf.keras.layers.Layer):
    def __init__(self, scaling, **kwargs):
        self.scaling = scaling
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.scaling:
            self.scale = self.add_weight(name='scale',
                                         shape=[input_shape[-1], ],
                                         initializer=tf.initializers.zeros,
                                         trainable=True)
        super(L2Normalization, self).build(input_shape)

    def call(self, inputs):
        output = tf.nn.l2_normalize(inputs, axis=-1)
        if self.scaling:
            output = tf.multiply(output, self.scale)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class BatchNormalization(tf.keras.layers.Layer):
    def __init__(self, decay=0.9, **kwargs):
        self.decay = decay
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=[input_shape[-1], ],
                                     initializer=tf.initializers.ones,
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=[input_shape[-1], ],
                                    initializer=tf.initializers.zeros,
                                    trainable=True)
        self.moving_mean = self.add_weight(name='moving_mean',
                                           shape=[input_shape[-1], ],
                                           initializer=tf.initializers.zeros,
                                           trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
                                               shape=[input_shape[-1], ],
                                               initializer=tf.initializers.ones,
                                               trainable=False)
        super(BatchNormalization, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        """
        variable = variable * decay + value * (1 - decay)
        """
        delta = variable * self.decay + value * (1 - self.decay)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, train):
        if train:
            # Here need tf.Variable.assign() and self.update()
            batch_mean, batch_variance = tf.nn.moments(inputs, list(range(len(inputs.shape) - 1)))
            mean_update = self.assign_moving_average(self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(self.moving_variance, batch_variance)
            self.add_update(mean_update, inputs=True)
            self.add_update(variance_update, inputs=True)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = tf.nn.batch_normalization(inputs, mean=mean, variance=variance, offset=self.beta, scale=self.gamma, variance_epsilon=1e-5)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class ConvBlock2D(tf.keras.layers.Layer):
    """
    Recursively create layer
    Usually convolution layer is followed by batch normalization, activation, pooling and dropout layers in order.
    Add parameters if need, such as max-pooling padding.
    """
    def __init__(self,
                 output_dim,
                 kernel=(3, 3),
                 strides=(1, 1, 1, 1),
                 padding='SAME',
                 use_activation=True,
                 activation=tf.nn.relu,
                 use_batch_normalization=True,
                 use_pooling=False,
                 pooling_size=(2, 2),
                 use_dropout=False,
                 dropout_rate=0.2,
                 **kwargs):
        super(ConvBlock2D, self).__init__(**kwargs)
        self.strides = strides
        self.padding = padding
        self.use_activation = use_activation
        self.activation = activation
        self.use_batch_normalization = use_batch_normalization
        self.use_pooling = use_pooling
        self.pooling_size = pooling_size
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        if self.use_batch_normalization:
            self.conv = Conv2D(output_dim=output_dim, kernel=kernel, use_bias=False, strides=self.strides, padding=self.padding)
            self.bn = BatchNormalization()
        else:
            self.conv = Conv2D(output_dim=output_dim, kernel=kernel, strides=self.strides, padding=self.padding)

    def call(self, inputs, train):
        net = self.conv(inputs)
        if self.use_batch_normalization:
            net = self.bn(net, train)
        if self.use_activation:
            net = self.activation(net)
        if self.use_pooling:
            net = tf.nn.max_pool2d(net, ksize=self.pooling_size, strides=self.pooling_size, padding='SAME')
        if self.use_dropout:
            if train:
                net = tf.nn.dropout(net, self.dropout_rate)
        return net