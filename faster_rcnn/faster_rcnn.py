import tensorflow as tf


class VGG(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(VGG, self).__init__(**kwargs)

    def call(self, inputs):
        """
        VGG网络提取特征，输入1000*600，经过4次pool后得到40*60的feature map
        """
        output = inputs
        return output

class RPN(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(RPN, self).__init__(**kwargs)

    def call(self, inputs):
        """
        RPN分两个分支，一条得到得到m*n*k*2个二分类信息，另一条得到m*n*k*4个坐标信息
        m*n是feature map大小，k个anchor数量，m=40，n=60，k=9
        """
        bbox = inputs
        cls = inputs
        return bbox, cls


class FullyConnect(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(FullyConnect, self).__init__(**kwargs)

    def call(self, inputs):
        """
        ROI pooling后得到batch*7*7*512的特征图，全连接处理，同样是两分支，类别预测和位置预测
        """
        bbox = inputs
        cls = inputs
        return bbox, cls

class Model(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.VGG = VGG()

    def call(self, inputs):

        feature_map = self.VGG(inputs)  # batch * 40 * 60 * 256

        return feature_map
