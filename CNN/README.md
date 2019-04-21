### CNN model

#### Basic keras style

In [CNN_keras_basic.py](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/CNN_keras_basic.py), CNN are implemented by basic keras style. 

`tf.config.gpu.set_per_process_memory_growth(enabled=True)` is to set the occupancy of GPU memory increases on demand. It is equal to
```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
```

#### Reimplement the basic function
In [CNN_keras_reimplement.py](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/CNN_keras_reimplement.py), FC layer, Conv2D layer, CrossEntropy and Accuracy function are reimplemented. No longer use `models.Sequential().add()` and the labels are convert to 'one hot' format.

New layers are inherited from `tf.keras.layers.Layer`. Core functions of it are `__init__()`, `build()` and `call()`. 


#### Reimplement the train function
In [CNN_train_reimplement.py](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/CNN_train_reimplement.py), `model.fit()` is not used. And the code show how to divide data, forward compute and update parameters. In `train` function, `@tf.function` is need to optimize the compute speed.
```
    with tf.GradientTape() as tape:
        prediction = model(x)
        loss = CrossEntropy(y, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```
The above code is the basic computational content required by the training process. Do not add unnecessary code such as dividing batch or transform the data format. 

In addition, Data types must match. If the input data is float64, the code can't run with error
```
TypeError: Input 'filter' of 'Conv2D' Op has type float32 that does not match type float64 of argument 'input'.
```
Add `astype(np.float32)` before input or add `tf.keras.backend.set_floatx('float64')` at the beginning of the code. However the second way takas more time in training.


#### Reimplement the batch normalization layer
Batch normalization layer is reimplemented in [BatchNormalization.py](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/BatchNormalization.py). It has different behaviors in training and test. In training it use batch data while it use moving data in test. However, it is incorrect to update the moving data by `=`, as follows
```
self.moving_mean = self.moving_mean * self.decay + batch_mean * (1 - self.decay)
self.moving_variance = self.moving_variance * self.decay + batch_variance * (1 - self.decay)
```
You need the `tf.Variable.assign()` and `tf.keras.models.Model.update` to implement.
```
mean_update = self.assign_moving_average(self.moving_mean, batch_mean)
variance_update = self.assign_moving_average(self.moving_variance, batch_variance)
self.add_update(mean_update, inputs=True)
self.add_update(variance_update, inputs=True)
```

#### VGG-16 model
In [VGG.py](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/VGG.py), the classical VGG-16 is implemented. Usually convolution layer is followed by batch normalization, activation, pooling and dropout layers in order. So they are combined into `ConvBlock2D`. 

To specify a different behavior in training and test, the subclass `Model` is needed, which have a `train` argument in `call()`. However, Model.summary() doesn't work. Just don't know why. The `output shape` is `multiple` all. You can add `print(net.shape)` in `call()` to show it.