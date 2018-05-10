# Metrics

## Problem of accuracy

[Jason Brownlee](https://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/):

> Consider, for interest, the problem of screening for a relatively rare condition such as cervical cancer, which has a prevalence of about 10% \(actual stats\). If a lazy Pap smear screener was to classify every slide they see as “normal”, they would have a 90% accuracy. Very impressive! But that figure completely ignores the fact that the 10% of women who do have the disease have not been diagnosed at all.

[7 Important Model Evaluation Error Metrics Everyone should know](https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/)

[Choosing the Right Metric for Evaluating ML Models — Part 1](https://towardsdatascience.com/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4)

> \[Log Loss\] looks at the probabilities themselves and not just the order of the predictions like AUC.

## F1

![{\displaystyle F\_{1}={\frac {2}{{\tfrac {1}{\mathrm {recall} }}+{\tfrac {1}{\mathrm {precision} }}}}=2\cdot {\frac {\mathrm {precision} \cdot \mathrm {recall} }{\mathrm {precision} +\mathrm {recall} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5663ca95d471868169c4e4ea57c936f1b6f4a588)

## R2

The [`r2_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score) function computes R², the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination). It provides a measure of how well future samples are likely to be predicted by the model. Best possible score is 1.0 and it can be negative \(because the model can be arbitrarily worse\). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

If ![\hat{y}\_i](http://scikit-learn.org/stable/_images/math/112652306646f689de7cf20153b2d70601aec3e1.png) is the predicted value of the ![i](http://scikit-learn.org/stable/_images/math/df0deb143e5ac127f00bd248ee8001ecae572adc.png)-th sample and ![y\_i](http://scikit-learn.org/stable/_images/math/07f6018e00c747406442bb3912e0209766fc9090.png) is the corresponding true value, then the score R² estimated over ![n\_{\text{samples}}](http://scikit-learn.org/stable/_images/math/d733c4bbf4bf946394a40154c6a82f0f936b6e58.png) is defined as 

![](http://scikit-learn.org/stable/_images/math/bdab7d608c772b3e382e2822a73ef557c80fbca2.png)

where ![\bar{y} =  \frac{1}{n\_{\text{samples}}} \sum\_{i=0}^{n\_{\text{samples}} - 1} y\_i](http://scikit-learn.org/stable/_images/math/4b4e8ee0c1363ed7f781ed3a12073cfd169e3f79.png).

This is the default scoring method for regression learners in scikit-learn.



## Log Loss

[Making Sense of Logarithmic Loss - datawookie](https://datawookie.netlify.com/blog/2015/12/making-sense-of-logarithmic-loss/)

[3.3. Model evaluation: quantifying the quality of predictions — scikit-learn 0.19.1 documentation](http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss)

[What is Log Loss? \| Kaggle](https://www.kaggle.com/dansbecker/what-is-log-loss/notebook)

