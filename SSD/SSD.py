import tensorflow as tf
import numpy as np
from Layers import ConvBlock2D, Conv2D


class Model(tf.keras.models.Model):
    """
    By this subclass, you can use to specify a different behavior in training and test such as dropout and batch
    normalization. However, model.summary() doesn't work. Just don't know why. You can add `print(net.shape)` in
    call() to show the output size of layers.
    """
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.conv1_1 = ConvBlock2D(64)
        self.conv1_2 = ConvBlock2D(64, use_pooling=True)
        self.conv2_1 = ConvBlock2D(128)
        self.conv2_2 = ConvBlock2D(128, use_pooling=True)
        self.conv3_1 = ConvBlock2D(256)
        self.conv3_2 = ConvBlock2D(256)
        self.conv3_3 = ConvBlock2D(256, use_pooling=True)
        self.conv4_1 = ConvBlock2D(512)
        self.conv4_2 = ConvBlock2D(512)
        self.conv4_3 = ConvBlock2D(512) # output before pooling
        self.conv5_1 = ConvBlock2D(512)
        self.conv5_2 = ConvBlock2D(512)
        self.conv5_3 = ConvBlock2D(512)
        self.conv6 = ConvBlock2D(1024)
        self.conv7 = ConvBlock2D(1024, kernel=(1, 1)) # output
        self.conv8_1 = ConvBlock2D(256, kernel=(1, 1))
        self.conv8_2 = ConvBlock2D(512) # output
        self.conv9_1 = ConvBlock2D(128, kernel=(1, 1))
        self.conv9_2 = ConvBlock2D(256) # output
        self.conv10_1 = ConvBlock2D(128, kernel=(1, 1))
        self.conv10_2 = ConvBlock2D(256) # output
        self.conv11_1 = ConvBlock2D(128, kernel=(1, 1))
        self.conv11_2 = ConvBlock2D(256) # output

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
        net = self.conv4_3(net, train) # output
        net = tf.nn.max_pool2d(net, (2, 2), (2, 2), 'SAME')
        net = self.conv5_1(net, train)
        net = self.conv5_2(net, train)
        net = self.conv5_3(net, train)
        # print(net.shape)
        net = self.conv6(net, train)
        net = self.conv7(net, train)

        return net