import tensorflow as tf
import numpy as np
import time
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

    def __init__(self, output_dim, activation=tf.nn.relu, use_bias=True, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
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
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

class Conv2D(tf.keras.layers.Layer):

    def __init__(self, output_dim, kernel=(3, 3), activation=tf.nn.relu,
                 use_bias=True, strides=(1, 1, 1, 1), padding='VALID', **kwargs):
        self.output_dim = output_dim
        self.kernel = kernel
        self.activation = activation
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
        output = self.activation(output)
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

@tf.function
def train(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(x)
        loss = CrossEntropy(y, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

@tf.function
def test(model, x, y):
    prediction = model(x)
    loss = CrossEntropy(y, prediction)
    return loss, prediction

if __name__ == '__main__':
    # tf.keras.backend.set_floatx('float64')
    tf.config.gpu.set_per_process_memory_growth(enabled=True)  # gpu memory set
    (train_images, train_labels, test_images, test_labels) = load_data('MNIST')

    with tf.device('/gpu:1'):  # If no GPU, comment on this line
        input = layers.Input(shape=(28, 28, 1))
        net = Conv2D(output_dim=32)(input)
        net = layers.MaxPooling2D((2, 2))(net)
        net = Conv2D(output_dim=64)(net)
        net = layers.MaxPooling2D((2, 2))(net)
        net = Conv2D(output_dim=64)(net)
        net = layers.Flatten()(net)
        net = FullyConnect(output_dim=64)(net)
        output = FullyConnect(output_dim=10, activation=tf.nn.softmax)(net)
        model = models.Model(inputs=input, outputs=output)
        # show
        model.summary()



        # train
        epoch = 5
        optimizer = tf.keras.optimizers.Adam()
        train_images = train_images.reshape((1875, 32, 28, 28, 1)).astype(np.float32)
        train_labels = train_labels.reshape((1875, 32, 10)).astype(np.float32)
        # TODO update the data input by tf.data

        for epoch_num in range(epoch):
            sum_loss = 0
            sum_num = 0
            cnt = 0
            start_time = time.time()
            for x, y in zip(train_images, train_labels):
                loss, prediction = train(model, x, y, optimizer)
                correct_num = Accuracy(y, prediction) * 32
                sum_loss += loss
                sum_num += correct_num

                cnt += 1
                if cnt % 100 == 0:
                    print('%d/%d, loss:%f, accuracy:%f'%(cnt, 1875, sum_loss/cnt/32, sum_num/cnt/32))
            end_time = time.time()
            print('epoch:%d, time:%.2f, loss:%f, accuracy:%f' %
                  (epoch_num, end_time-start_time, sum_loss/60000, sum_num/60000))

        # test
        test_images = test_images.reshape((625, 16, 28, 28, 1)).astype(np.float32)
        test_labels = test_labels.reshape((625, 16, 10)).astype(np.float32)

        sum_loss = 0
        sum_num = 0
        cnt = 0
        start_time = time.time()
        for x, y in zip(test_images, test_labels):
            loss, prediction = test(model, x, y)
            correct_num = Accuracy(y, prediction) * 16
            sum_loss += loss
            sum_num += correct_num

            cnt += 1
            if cnt % 100 == 0:
                print('%d/%d, loss:%f, accuracy:%f' % (cnt, 625, sum_loss / cnt / 16, sum_num / cnt / 16))
        end_time = time.time()
        print('test, time:%.2f, loss:%f, accuracy:%f' %
              (end_time - start_time, sum_loss / 10000, sum_num / 10000))
