# Feature Importance

Even [Jeremy ](https://youtu.be/0v93qHDqq_g?t=26m26s)doesn't care about the unit of feature importance.

[Jeremy's method ](https://youtu.be/0v93qHDqq_g?t=26m26s)in finding out the top features:

* Train a model using all the features
* Train a model using the top X features so that the metrics are not worst than the 1st model. Select the X features using the threshold of feature important 0.0005 as the starting point. If performance gets worse, try to use a lower threshold.

The reason why we can't only rely on the feature importance from the 1st model is that correlated features would shared the importance so the its importance is lower than its actual importance.

