import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

def load_data(PATH):
    train_images = np.load(PATH + '/x_train.npy')
    train_labels = np.load(PATH + '/y_train.npy')
    test_images = np.load(PATH + '/x_test.npy')
    test_labels = np.load(PATH + '/y_test.npy')

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    # mormalize
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # one hot
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    return train_images, train_labels, test_images, test_labels

class FullyConnect(tf.keras.layers.Layer):

    def __init__(self, output_dim, use_bias=True, **kwargs):
        self.output_dim = output_dim
        self.use_bias = use_bias
        super(FullyConnect, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[-1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='kernel',
                                          shape=[self.output_dim,],
                                          initializer='zeros',
                                          trainable=True)
        super(FullyConnect, self).build(input_shape)

    def call(self, inputs):
        if self.use_bias:
            output = tf.add(tf.matmul(inputs, self.kernel), self.bias)
        else:
            output = tf.matmul(inputs, self.kernel)
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

class Conv2D(tf.keras.layers.Layer):

    def __init__(self, output_dim, kernel=(3, 3), use_bias=True, strides=(1, 1, 1, 1), padding='VALID', **kwargs):
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
                                      initializer='normal',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='kernel',
                                        shape=[self.output_dim,],
                                        initializer='zeros',
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

def CrossEntropy(y_true, y_pred):
    cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred))
    return cross_entropy

def Accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return accuracy

if __name__ == '__main__':
    tf.config.gpu.set_per_process_memory_growth(enabled=True)  # gpu memory set
    (train_images, train_labels, test_images, test_labels) = load_data('MNIST')

    with tf.device('/gpu:1'):  # If no GPU, comment on this line
        input = layers.Input(shape=(28, 28, 1))
        net = Conv2D(output_dim=32)(input)
        net = tf.nn.relu(net)
        net = layers.MaxPooling2D((2, 2))(net)
        net = Conv2D(output_dim=64)(net)
        net = tf.nn.relu(net)
        net = layers.MaxPooling2D((2, 2))(net)
        net = Conv2D(output_dim=64)(net)
        net = tf.nn.relu(net)
        net = layers.Flatten()(net)
        net = FullyConnect(output_dim=64)(net)
        net = tf.nn.relu(net)
        net = FullyConnect(output_dim=10)(net)
        output = tf.nn.softmax(net)
        model = models.Model(inputs=input, outputs=output)
        # show
        model.summary()
        # train
        model.compile(optimizer='adam', loss=CrossEntropy, metrics=[Accuracy])
        model.fit(train_images, train_labels, batch_size=32, epochs=5)
        # test
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(test_acc)
