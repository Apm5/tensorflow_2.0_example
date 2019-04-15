###Realization of CNN in different styles

In CNN_keras_basic.py, CNN are implemented by basic keras style. 

`tf.config.gpu.set_per_process_memory_growth(enabled=True)` is to set the occupancy of GPU memory increases on demand. It is equal to
```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
```

