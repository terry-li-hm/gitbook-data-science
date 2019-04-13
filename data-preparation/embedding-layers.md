# Embedding Layers

[This ](https://www.kaggle.com/colinmorris/embedding-layers)is a very good tutorial from Kaggle. In short, it is the state-of-the-art way to handle categorical features with lots of possible values \(high _cardinality_\). However, it seems it is only applicable to neural networks.

The following is a technical tips on which optimizer to use per the tutorial:

> ```python
> model.compile(
>     # Technical note: when using embedding layers, I highly recommend using one of the optimizers
>     # found  in tf.train: https://www.tensorflow.org/api_guides/python/train#Optimizers
>     # Passing in a string like 'adam' or 'SGD' will load one of keras's optimizers (found under 
>     # tf.keras.optimizers). They seem to be much slower on problems like this, because they
>     # don't efficiently handle sparse gradient updates.
>     tf.train.AdamOptimizer(0.005),
>     loss='MSE',
>     metrics=['MAE'],
> )
> ```

