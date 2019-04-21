import tensorflow as tf
import numpy as np
import pickle as p
import time
import os
from tensorflow.keras import layers, models

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR(Foldername):
    train_data = np.zeros([50000, 32, 32, 3])
    train_label = np.zeros([50000, 10])
    for sample in range(5):
        X, Y = load_CIFAR_batch(Foldername + "/data_batch_" + str(sample + 1))

        for i in range(3):
            train_data[10000 * sample:10000 * (sample + 1), :, :, i] = X[:, i, :, :]
        for i in range(10000):
            train_label[i + 10000 * sample][Y[i]] = 1

    test_data = np.zeros([10000, 32, 32, 3])
    test_label = np.zeros([10000, 10])
    X, Y = load_CIFAR_batch(Foldername + "/test_batch")
    for i in range(3):
        test_data[0:10000, :, :, i] = X[:, i, :, :]
    for i in range(10000):
        test_label[i][Y[i]] = 1

    return train_data, train_label, test_data, test_label

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
                 use_activation=True,
                 activation=tf.nn.relu,
                 use_batch_normalization=True,
                 use_pooling=True,
                 pooling_size=(2, 2),
                 use_dropout=False,
                 dropout_rate=0.2,
                 **kwargs):
        super(ConvBlock2D, self).__init__(**kwargs)
        self.use_activation = use_activation
        self.activation = activation
        self.use_batch_normalization = use_batch_normalization
        self.use_pooling = use_pooling
        self.pooling_size = pooling_size
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        if self.use_batch_normalization:
            self.conv = Conv2D(output_dim=output_dim, use_bias=False)
            self.bn = BatchNormalization()
        else:
            self.conv = Conv2D(output_dim=output_dim)

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


class Model(tf.keras.models.Model):
    """
    By this subclass, you can use to specify a different behavior in training and test such as dropout and batch
    normalization. However, model.summary() doesn't work. Just don't know why. You can add `print(net.shape)` in
    call() to show the output size of layers.
    """
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.conv1_1 = ConvBlock2D(64, use_pooling=False)
        self.conv1_2 = ConvBlock2D(64, use_dropout=True)
        self.conv2_1 = ConvBlock2D(128, use_pooling=False)
        self.conv2_2 = ConvBlock2D(128, use_dropout=True)
        self.conv3_1 = ConvBlock2D(256, use_pooling=False)
        self.conv3_2 = ConvBlock2D(256, use_pooling=False)
        self.conv3_3 = ConvBlock2D(256, use_dropout=True)
        self.conv4_1 = ConvBlock2D(512, use_pooling=False)
        self.conv4_2 = ConvBlock2D(512, use_pooling=False)
        self.conv4_3 = ConvBlock2D(512, use_dropout=True)
        self.conv5_1 = ConvBlock2D(512, use_pooling=False)
        self.conv5_2 = ConvBlock2D(512, use_pooling=False)
        self.conv5_3 = ConvBlock2D(512, use_dropout=True)
        self.fc1 = FullyConnect(output_dim=64)
        self.fc2 = FullyConnect(output_dim=10, activation=tf.nn.softmax)

    def call(self, inputs, train):
        net = self.conv1_1(inputs, train)
        net = self.conv1_2(net, train)
        net = self.conv2_1(net, train)
        net = self.conv2_2(net, train)
        net = self.conv3_1(net, train)
        net = self.conv3_2(net, train)
        net = self.conv3_3(net, train)
        net = self.conv4_1(net, train)
        net = self.conv4_2(net, train)
        net = self.conv4_3(net, train)
        net = self.conv5_1(net, train)
        net = self.conv5_2(net, train)
        net = self.conv5_3(net, train)
        # print(net.shape)
        net = tf.reshape(net, [net.shape[0], -1])  # Flatten
        net = self.fc1(net)
        net = self.fc2(net)
        return net

def CrossEntropy(y_true, y_pred):
    cross_entropy = -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
    return cross_entropy

def Accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return accuracy

@tf.function
def train(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(x, train=True)
        loss = CrossEntropy(y, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

@tf.function
def test(model, x, y):
    prediction = model(x, train=False)
    loss = CrossEntropy(y, prediction)
    return loss, prediction

if __name__ == '__main__':
    # tf.keras.backend.set_floatx('float64')
    tf.config.gpu.set_per_process_memory_growth(enabled=True)  # gpu memory set
    (train_images, train_labels, test_images, test_labels) = load_CIFAR('/home/user/Documents/dataset/Cifar-10')

    with tf.device('/gpu:0'):  # If no GPU, comment on this line
        model = Model()

        epoch = 30
        optimizer = tf.keras.optimizers.Adam()
        train_images = train_images.reshape((1000, 50, 32, 32, 3)).astype(np.float32)
        train_labels = train_labels.reshape((1000, 50, 10)).astype(np.float32)
        # TODO update the data input by tf.data

        for epoch_num in range(epoch):
            # train
            sum_loss = 0
            sum_num = 0
            cnt = 0
            start_time = time.time()
            for x, y in zip(train_images, train_labels):
                loss, prediction = train(model, x, y, optimizer)
                correct_num = Accuracy(y, prediction) * 50
                sum_loss += loss
                sum_num += correct_num

                cnt += 1
                if cnt % 100 == 0:
                    print('%d/%d, loss:%f, accuracy:%f'%(cnt, 1000, sum_loss/cnt/50, sum_num/cnt/50))
            end_time = time.time()
            print('epoch:%d, time:%.2f, loss:%f, accuracy:%f' %
                  (epoch_num, end_time-start_time, sum_loss/50000, sum_num/50000))

            # test
            test_images = test_images.reshape((200, 50, 32, 32, 3)).astype(np.float32)
            test_labels = test_labels.reshape((200, 50, 10)).astype(np.float32)

            sum_loss = 0
            sum_num = 0
            cnt = 0
            start_time = time.time()
            for x, y in zip(test_images, test_labels):
                loss, prediction = test(model, x, y)
                correct_num = Accuracy(y, prediction) * 50
                sum_loss += loss
                sum_num += correct_num

                cnt += 1
                if cnt % 100 == 0:
                    print('%d/%d, loss:%f, accuracy:%f' % (cnt, 200, sum_loss / cnt / 50, sum_num / cnt / 50))
            end_time = time.time()
            print('test, time:%.2f, loss:%f, accuracy:%f' %
                  (end_time - start_time, sum_loss / 10000, sum_num / 10000))
