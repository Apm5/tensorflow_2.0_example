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