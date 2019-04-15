### Realization of CNN in different styles

In [CNN_keras_basic.py](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/CNN_keras_basic.py), CNN are implemented by basic keras style. 

`tf.config.gpu.set_per_process_memory_growth(enabled=True)` is to set the occupancy of GPU memory increases on demand. It is equal to
```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
```

In [CNN_keras_reimplement.py](https://github.com/Apm5/tensorflow_2.0_example/blob/master/CNN/CNN_keras_reimplement.py), FC layer, Conv2D layer, CrossEntropy and Accuracy function are reimplemented. No longer use `models.Sequential().add()` and the labels are convert to 'one hot' format.

New layers are inherited from `tf.keras.layers.Layer`. Core functions of it are `__init__()`, `build()` and `call()`. 