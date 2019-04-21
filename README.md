### This is an example for tensorflow 2.0-alpha

In version 2.0, many features of tensorflow are deleted or changed, such as `tf.Session`. And in new version, Keras became the core function, which makes the code style quite different from before.

In this project, I will implement various layers and models. And I will try not to rely on advanced APIs to make it easy to change the details of the model.

Models and layers are as follows:

#### DONE:

[Reimplement fully connected layer](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/CNN_keras_reimplement.py)

[Reimplement convolution layer](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/CNN_keras_reimplement.py)

[Use unofficial evaluation and loss function](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/CNN_train_reimplement.py)

[Use unofficial train function](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/CNN_train_reimplement.py)

[Batch normalization](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/BatchNormalization.py)

[VGG-16](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/VGG.py)

#### TODO:

tf.data

Gabor convolution

LSTM

ResNet

RCNN series

......