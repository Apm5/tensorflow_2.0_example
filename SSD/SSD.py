import tensorflow as tf
import numpy as np
import config
import time
import cv2
from Layers import ConvBlock2D, Conv2D, DilatedConv2D, L2Normalization
from load_data import load_data, generate_default_boxes

class Model(tf.keras.models.Model):
    """
    By this subclass, you can use to specify a different behavior in training and test such as dropout and batch
    normalization. However, model.summary() doesn't work. Just don't know why. You can add `print(net.shape)` in
    call() to show the output size of layers.
    """
    def __init__(self, class_num=config.class_num, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.class_num = class_num
        self.conv1_1 = Conv2D(64)
        self.conv1_2 = Conv2D(64)
        self.conv2_1 = Conv2D(128)
        self.conv2_2 = Conv2D(128)
        self.conv3_1 = Conv2D(256)
        self.conv3_2 = Conv2D(256)
        self.conv3_3 = Conv2D(256)
        self.conv4_1 = Conv2D(512)
        self.conv4_2 = Conv2D(512)
        self.conv4_3 = Conv2D(512)
        self.l2_normalization = L2Normalization(scaling=True)
        self.conv5_1 = Conv2D(512)
        self.conv5_2 = Conv2D(512)
        self.conv5_3 = Conv2D(512)
        self.conv6 = DilatedConv2D(1024, rate=2)
        self.conv7 = Conv2D(1024, kernel=(1, 1))
        self.conv8_1 = Conv2D(256, kernel=(1, 1))
        self.conv8_2 = Conv2D(512, strides=(1, 2, 2, 1))
        self.conv9_1 = Conv2D(128, kernel=(1, 1))
        self.conv9_2 = Conv2D(256, strides=(1, 2, 2, 1))
        self.conv10_1 = Conv2D(128, kernel=(1, 1))
        self.conv10_2 = Conv2D(256, strides=(1, 2, 2, 1))
        self.conv11_1 = Conv2D(128, kernel=(1, 1))
        self.conv11_2 = Conv2D(256, padding='VALID')

        self.conv_cls = [Conv2D(box_num * self.class_num) for box_num in config.box_num]
        self.conv_loc = [Conv2D(box_num * 4) for box_num in config.box_num]


    def call(self, inputs, train):
        feature_map_list = []
        batch_size = inputs.shape[0]

        net = self.conv1_1(inputs)
        net = tf.nn.relu(net)
        net = self.conv1_2(net)
        net = tf.nn.relu(net)
        net = tf.nn.max_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')

        net = self.conv2_1(net)
        net = tf.nn.relu(net)
        net = self.conv2_2(net)
        net = tf.nn.relu(net)
        net = tf.nn.max_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')

        net = self.conv3_1(net)
        net = tf.nn.relu(net)
        net = self.conv3_2(net)
        net = tf.nn.relu(net)
        net = self.conv3_3(net)
        net = tf.nn.relu(net)
        net = tf.nn.max_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')

        net = self.conv4_1(net)
        net = tf.nn.relu(net)
        net = self.conv4_2(net)
        net = tf.nn.relu(net)
        net = self.conv4_3(net) # output
        net = tf.nn.relu(net)
        feature_map_list.append(self.l2_normalization(net))
        print(net.shape)
        net = tf.nn.max_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')

        net = self.conv5_1(net)
        net = tf.nn.relu(net)
        net = self.conv5_2(net)
        net = tf.nn.relu(net)
        net = self.conv5_3(net)
        net = tf.nn.relu(net)
        net = tf.nn.max_pool2d(net, ksize=(3, 3), strides=(1, 1), padding='SAME')

        net = self.conv6(net)
        net = tf.nn.relu(net)
        # if train:
        #     net = tf.nn.dropout(net, 0.5)
        net = self.conv7(net)
        net = tf.nn.relu(net)
        feature_map_list.append(net)
        # if train:
        #     net = tf.nn.dropout(net, 0.5)
        print(net.shape)

        net = self.conv8_1(net)
        net = tf.nn.relu(net)
        net = self.conv8_2(net)
        net = tf.nn.relu(net)
        feature_map_list.append(net)
        print(net.shape)

        net = self.conv9_1(net)
        net = tf.nn.relu(net)
        net = self.conv9_2(net)
        net = tf.nn.relu(net)
        feature_map_list.append(net)
        print(net.shape)

        net = self.conv10_1(net)
        net = tf.nn.relu(net)
        net = self.conv10_2(net)
        net = tf.nn.relu(net)
        feature_map_list.append(net)
        print(net.shape)
        tf.nn.l2_normalize
        tf.nn.atrous_conv2d
        net = self.conv11_1(net)
        net = tf.nn.relu(net)
        net = self.conv11_2(net)
        net = tf.nn.relu(net)
        feature_map_list.append(net)
        print(net.shape)

        cls_list = []
        loc_list = []
        for i, feature_map in enumerate(feature_map_list):
            cls_list.append(tf.reshape(self.conv_cls[i](feature_map), [batch_size, -1, self.class_num]))
            loc_list.append(tf.reshape(self.conv_loc[i](feature_map), [batch_size, -1, 4]))
        cls = tf.concat(cls_list, axis=1)
        cls = tf.nn.softmax(cls)
        loc = tf.concat(loc_list, axis=1)
        # print(cls.shape, loc.shape)

        # tf.nn.top_k
        #         tf.logical_and
        return cls, loc

class Model_batchnorm(tf.keras.models.Model):
    """
    By this subclass, you can use to specify a different behavior in training and test such as dropout and batch
    normalization. However, model.summary() doesn't work. Just don't know why. You can add `print(net.shape)` in
    call() to show the output size of layers.
    """
    def __init__(self, class_num=config.class_num, **kwargs):
        super(Model_batchnorm, self).__init__(**kwargs)
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
        self.conv8_2 = ConvBlock2D(512, strides=(1, 2, 2, 1)) # output
        self.conv9_1 = ConvBlock2D(128, kernel=(1, 1))
        self.conv9_2 = ConvBlock2D(256, strides=(1, 2, 2, 1)) # output
        self.conv10_1 = ConvBlock2D(128, kernel=(1, 1))
        self.conv10_2 = ConvBlock2D(256, strides=(1, 2, 2, 1)) # output
        self.conv11_1 = ConvBlock2D(128, kernel=(1, 1))
        self.conv11_2 = ConvBlock2D(256, padding='VALID') # output

        self.conv_cls = [Conv2D(box_num * self.class_num) for box_num in config.box_num]
        self.conv_loc = [Conv2D(box_num * 4) for box_num in config.box_num]


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
        # print(net.shape)
        net = self.conv6(net, train)
        net = self.conv7(net, train)
        feature_map_list.append(net)
        print(net.shape)
        net = self.conv8_1(net, train)
        net = self.conv8_2(net, train)
        feature_map_list.append(net)
        print(net.shape)
        net = self.conv9_1(net, train)
        net = self.conv9_2(net, train)
        feature_map_list.append(net)
        print(net.shape)
        net = self.conv10_1(net, train)
        net = self.conv10_2(net, train)
        feature_map_list.append(net)
        print(net.shape)
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
        cls = tf.nn.softmax(cls)
        loc = tf.concat(loc_list, axis=1)
        # print(cls.shape, loc.shape)

        # tf.nn.top_k
        #         tf.logical_and
        return cls, loc

def cls_loss(cls_true, cls_pred):
    def cross_entropy(y_true, y_pred):
        loss = -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))
        return loss

    loss = tf.cast(0.0, dtype=tf.float32)
    # print(loss)

    # positive
    ind_pos = tf.where(tf.less(cls_true, 20))
    print(ind_pos)
    cls_true_pos = tf.gather_nd(cls_true, ind_pos)
    cls_true_pos = tf.one_hot(cls_true_pos, 21)
    # print(cls_true_pos)
    num_pos = tf.shape(cls_true_pos)[0]
    cls_pred_pos = tf.gather_nd(cls_pred, ind_pos)
    # print(cls_pred_pos)
    loss = loss + cross_entropy(cls_true_pos, cls_pred_pos)
    # print(loss)

    # negative
    ind_neg = tf.where(tf.equal(cls_true, 20))
    cls_pred_neg = tf.gather_nd(cls_pred, ind_neg)
    neg = tf.nn.top_k(-cls_pred_neg[:, 20], num_pos * 3)  # '-' for top-k minimum
    loss = loss - tf.reduce_sum(tf.math.log(tf.clip_by_value(-neg[0], 1e-10, 1.0)))  # '-' is necessary
    # print(loss)

    # conf_pos_mask * tf.nn.softmax_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
    return loss

def loc_loss(cls_true, loc_true, loc_pred):
    # In fact, loc is the offset.

    def smooth_l1(x):
        if x < 1:
            output = 0.5 * x * x
        else:
            output = x - 0.5
        return output

    loss = tf.cast(0.0, dtype=tf.float32)

    ind = tf.where(tf.less(cls_true, 20))
    loc_true_ = tf.gather_nd(loc_true, ind)
    loc_pred_ = tf.gather_nd(loc_pred, ind)
    diff = tf.reshape(tf.math.abs(loc_true_ - loc_pred_), [-1])
    # loss = tf.reduce_sum(tf.map_fn(fn= lambda x: smooth_l1(x), elems=diff))  # fucking idiot function
    ind = tf.where(tf.less(diff, 1.0))
    loss += tf.reduce_sum(0.5 * tf.math.pow(tf.gather_nd(diff, ind), 2))
    ind = tf.where(tf.less(1.0, diff))
    loss += tf.reduce_sum(tf.gather_nd(diff, ind) - 0.5)

    return loss

def Loss(cls_true, cls_pred, loc_true, loc_pred):
    ind_pos = tf.where(tf.less(cls_true, 20))  # matched default boxes
    ind_neg = tf.where(tf.equal(cls_true, 20))  # back ground default boxes
    N = tf.shape(ind_pos)[0]

    # classification loss
    cls_true_pos = tf.gather_nd(cls_true, ind_pos)
    cls_true_pos = tf.one_hot(cls_true_pos, 21)
    cls_pred_pos = tf.gather_nd(cls_pred, ind_pos)

    cls_pred_neg = tf.gather_nd(cls_pred, ind_neg)
    neg_top_k = tf.nn.top_k(-cls_pred_neg[:, 20], N * 3)  # '-' for top-k minimum
    # print(tf.math.log(tf.clip_by_value(cls_pred_pos, 1e-10, 1.0)))
    # print(cls_true_pos * tf.math.log(tf.clip_by_value(cls_pred_pos, 1e-10, 1.0)))
    # print(tf.reduce_sum(cls_true_pos * tf.math.log(tf.clip_by_value(cls_pred_pos, 1e-10, 1.0))))
    # print(tf.reduce_sum(tf.math.log(tf.clip_by_value(-neg_top_k[0], 1e-10, 1.0))))
    print(neg_top_k)
    L_cls = - tf.reduce_sum(cls_true_pos * tf.math.log(tf.clip_by_value(cls_pred_pos, 1e-10, 1.0))) \
            - tf.reduce_sum(tf.math.log(tf.clip_by_value(-neg_top_k[0], 1e-10, 1.0)))  # '-' is necessary

    # location loss
    loc_true_ = tf.gather_nd(loc_true, ind_pos)
    loc_pred_ = tf.gather_nd(loc_pred, ind_pos)
    # smooth l1
    diff = tf.reshape(tf.math.abs(loc_true_ - loc_pred_), [-1])
    mask = tf.cast(tf.less(diff, 1.0), dtype=tf.float32)
    # print(diff, mask)
    L_loc = tf.reduce_sum(0.5 * tf.math.pow(diff, 2) * mask) + \
            tf.reduce_sum((diff - 0.5) * (1 - mask))

    N = tf.cast(N, dtype=tf.float32)
    L = (L_cls + L_loc) / N
    return L_cls / N, L_loc / N, L

def Loss_mask(cls_true, cls_pred, loc_true, loc_pred):
    pmask = tf.less(cls_true, 20)  # positive mask
    fpmask = tf.cast(pmask, dtype=tf.float32)
    nmask = tf.equal(cls_true, 20)  # negative mask
    p_num = tf.reduce_sum(tf.cast(pmask, dtype=tf.int32))  # positive number
    # print(p_num)
    cls_true_one_hot = tf.one_hot(cls_true, 21)

    b_cls_pred = cls_pred[:, :, 20] + fpmask  # back ground
    top_k = tf.nn.top_k(tf.reshape(-(b_cls_pred), [-1]), p_num * 3)
    # cls_pred[:, :, 20] for back ground pred [batch, 8732]
    # + fpmask to let object > 1.0
    # then - (cls_pred[:, :, 20] + fpmask) , object < -1.0
    # top_k is all back ground that match ground truth
    top_k_threshold = -top_k[0][-1]
    # top_k[0] is the value
    # -top_k[0][-1] is the largest one of top_k
    select_nmask = tf.less(b_cls_pred, top_k_threshold)
    # print(tf.reduce_sum(tf.cast(select_nmask, tf.float32)))
    nmask = tf.logical_and(nmask, select_nmask)
    # check
    # print(p_num)
    # print(tf.reduce_sum(tf.cast(select_nmask, dtype=tf.int32)))
    # print(tf.reduce_sum(tf.cast(nmask, dtype=tf.int32)))
    mask = tf.logical_or(pmask, nmask)
    fmask = tf.cast(mask, dtype=tf.float32)
    # print(tf.reduce_sum(tf.cast(nmask, tf.float32)))
    # print(tf.reduce_sum(tf.cast(mask, tf.float32)))

    cross_entropy = cls_true_one_hot * tf.math.log(tf.clip_by_value(cls_pred, 1e-10, 1.0))
    # print(cross_entropy.shape)
    # print(tf.reduce_sum(cross_entropy, axis=-1))
    # print(tf.reduce_sum(cross_entropy, axis=-1) * fmask)
    L_cls = - tf.reduce_sum(tf.reduce_sum(cross_entropy, axis=-1) * fmask)
    # print(l_cls)

    diff = tf.math.abs(loc_true - loc_pred)
    smooth_l1_mask = tf.cast(tf.less(diff, 1.0), dtype=tf.float32)
    # print(smooth_l1_mask.shape)
    smooth_l1 = tf.reduce_sum(0.5 * tf.math.pow(diff, 2) * smooth_l1_mask, axis=-1) + \
                tf.reduce_sum((diff - 0.5) * (1 - smooth_l1_mask), axis=-1)
    # print(smooth_l1.shape)
    L_loc = tf.reduce_sum(smooth_l1 * fpmask)
    # print(l_loc)

    p_num = tf.cast(p_num, dtype=tf.float32)
    L = (L_cls + L_loc) / p_num
    # L = L_cls / p_num
    # L = L_loc / p_num
    return L_cls / p_num, L_loc / p_num, L

@tf.function
def train(model, images, cls_true, loc_true, optimizer):
    with tf.GradientTape() as tape:
        cls_pred, loc_pred = model(images, train=True)  # 0.015s
        # c_loss = cls_loss(cls_true, cls_pred)  # 0.002s
        # l_loss = loc_loss(cls_true, loc_true, loc_pred)  # 0.25s
        # loss = c_loss + l_loss
        c_loss, l_loss, loss = Loss_mask(cls_true, cls_pred, loc_true, loc_pred)
        # TODO optimize the loss function
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return c_loss, l_loss, loss

if __name__ == '__main__':
    tf.config.gpu.set_per_process_memory_growth(enabled=True)  # gpu memory set
    # with open(config.train, 'r') as f:
    #     name_list = []
    #     for name in f:
    #         name_list.append(name[0:6])
    # print('train num:', len(name_list))
    # print(name_list[3:4])
    #
    # default_boxes = generate_default_boxes('ltrb')
    # images, loc_true, cls_true = load_data(config.path, name_list[3:4], default_boxes)
    # print(loc_true, cls_true)
    # model = Model()
    # test(model, images, cls_true, loc_true)
    with tf.device('/gpu:2'):
        model = Model_batchnorm()
        model.load_weights('weights/weights_499')
        # default_boxes = generate_default_boxes('ltrb')
        # x = np.random.rand(8, 300, 300, 3)
        # scores, boxes = model(x, train=False)
        # print(np.shape(scores), np.shape(boxes))
        # for var in model.variables:
        #     print(var.name)

        # f = open('SSD_result.txt', 'a')
        # optimizer = tf.keras.optimizers.Adam(0.001)


        images_list = np.load('preload/images.npy')
        cls_list = np.load('preload/cls.npy')
        loc_list = np.load('preload/loc.npy')
        sample_num = len(images_list)
        print(sample_num)

        epoch = 200
        batch_size = 32
        batch_num = int(sample_num / batch_size)
        shuffle_seed = np.arange(sample_num)

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)

        for epoch_ind in range(0, epoch):
            # train
            total_loss = 0
            total_c_loss = 0
            total_l_loss = 0
            np.random.shuffle(shuffle_seed)
            images_list = images_list[shuffle_seed]
            cls_list = cls_list[shuffle_seed]
            loc_list = loc_list[shuffle_seed]
            for i in range(batch_num):
                # images, loc_true, cls_true = load_data(config.path, name_list[i:i+batch_size], default_boxes)
                images, loc_true, cls_true = images_list[i:i+batch_size], loc_list[i:i+batch_size], cls_list[i:i+batch_size]
                # print(cls_true)

                # ind = np.where(cls_true < 20)
                # print(ind, np.shape(ind))
                c, l, loss = train(model, images, cls_true, loc_true, optimizer)
                total_loss += loss.numpy()
                total_c_loss += c.numpy()
                total_l_loss += l.numpy()
                print('epoch:', epoch_ind,
                      i, '/', batch_num,
                      'loss:', loss.numpy(),
                      'cls:', c.numpy(),
                      'loc:', l.numpy())
            if epoch_ind == 0:
                for var in model.variables:
                    print(var.name)
            if epoch_ind % 10 == 9:
                model.save_weights('weights/weights_'+str(epoch_ind))
            # f.write('epoch: %d, loss: %f, cls: %f, loc: %f' % (epoch_ind, total_loss, total_c_loss, total_l_loss) + '\n')
