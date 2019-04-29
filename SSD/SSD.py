import tensorflow as tf
import numpy as np
import config
from Layers import ConvBlock2D, Conv2D

def load_data(path, name_list):




    return images, labels

class Model(tf.keras.models.Model):
    """
    By this subclass, you can use to specify a different behavior in training and test such as dropout and batch
    normalization. However, model.summary() doesn't work. Just don't know why. You can add `print(net.shape)` in
    call() to show the output size of layers.
    """
    def __init__(self, class_num, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.class_num = class_num
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

        self.conv_cls = [Conv2D(box_num * self.class_num) for box_num in config.box_num]
        self.conv_loc = [Conv2D(box_num * 4) for box_num in config.box_num]

        self.scale = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9, 1.04] # maybe different from other implementation
        self.default_box_num = [4, 6, 6, 6, 4, 4]
        self.ratio = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]

    def call(self, inputs, train):
        feature_map_list = []
        batch_size = inputs.shape[0]

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
        feature_map_list.append(net)
        print(net.shape)
        net = tf.nn.max_pool2d(net, (2, 2), (2, 2), 'SAME')
        net = self.conv5_1(net, train)
        net = self.conv5_2(net, train)
        net = self.conv5_3(net, train)
        print(net.shape)
        net = self.conv6(net, train)
        net = self.conv7(net, train)
        feature_map_list.append(net)
        print(net.shape)
        net = tf.nn.max_pool2d(net, (2, 2), (2, 2), 'SAME')
        net = self.conv8_1(net, train)
        net = self.conv8_2(net, train)
        feature_map_list.append(net)
        print(net.shape)
        net = tf.nn.max_pool2d(net, (2, 2), (2, 2), 'SAME')
        net = self.conv9_1(net, train)
        net = self.conv9_2(net, train)
        feature_map_list.append(net)
        print(net.shape)
        net = tf.nn.max_pool2d(net, (2, 2), (2, 2), 'SAME')
        net = self.conv10_1(net, train)
        net = self.conv10_2(net, train)
        feature_map_list.append(net)
        print(net.shape)
        net = tf.nn.max_pool2d(net, (3, 3), (3, 3), 'SAME')
        net = self.conv11_1(net, train)
        net = self.conv11_2(net, train)
        feature_map_list.append(net)
        print(net.shape)

        cls_list = []
        loc_list = []
        for i, feature_map in enumerate(feature_map_list):
            cls_list.append(tf.reshape(self.conv_cls[i](feature_map), [batch_size, -1, self.class_num]))
            loc_list.append(tf.reshape(self.conv_loc[i](feature_map), [batch_size, -1, 4]))
        cls = tf.concat(cls_list, axis=1)
        loc = tf.concat(loc_list, axis=1)
        print(cls.shape, loc.shape)

        # tf.nn.top_k
        #         tf.logical_and
        return cls, loc

def cls_loss(cls_true, loc_pred):
    return True

def loc_loss(loc_true, loc_pred):
    return True


@tf.function
def train(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        cls, loc = model(images, train=True)
        loss = cls_loss(labels[:, :, 4], cls) + loc_loss(labels[:, :, 0:4], loc)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

if __name__ == '__main__':
    tf.config.gpu.set_per_process_memory_growth(enabled=True)  # gpu memory set
    model = Model(10)
    x = np.random.rand(4, 300, 300, 3)
    with tf.device('/gpu:1'):
        print(config.scale)
        scores, boxes = model(x, train=False)
        for var in model.variables:
            print(var.name)