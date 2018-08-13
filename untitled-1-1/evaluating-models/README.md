# Metrics

* [Introduction to Loss Functions](https://blog.algorithmia.com/introduction-to-loss-functions/)
* [Choosing the Right Metric for Evaluating Machine Learning Models — Part 2](https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428)
* [Receiver Operating Characteristic Curves Demystified \(in Python\)](https://towardsdatascience.com/receiver-operating-characteristic-curves-demystified-in-python-bd531a4364d0)
* [Simple guide to confusion matrix terminology](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
* [Evaluation Metrics for Classification – machinelearning-blog.com](https://machinelearning-blog.com/2018/04/03/evaluation-metrics-for-classification/)
* [Gini Coefficient - An Intuitive Explanation \| Kaggle](https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation)
* [Putting Accuracy in Context - Zillow Artificial Intelligence](https://www.zillow.com/data-science/putting-accuracy-context/)
* [Unified Cross-Platform Performance Metrics – Several People Are Coding](https://slack.engineering/unified-cross-platform-performance-metrics-adeb371a8814)



## Problem of accuracy

[Jason Brownlee](https://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/):

> Consider, for interest, the problem of screening for a relatively rare condition such as cervical cancer, which has a prevalence of about 10% \(actual stats\). If a lazy Pap smear screener was to classify every slide they see as “normal”, they would have a 90% accuracy. Very impressive! But that figure completely ignores the fact that the 10% of women who do have the disease have not been diagnosed at all.

[7 Important Model Evaluation Error Metrics Everyone should know](https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/)

[Choosing the Right Metric for Evaluating ML Models — Part 1](https://towardsdatascience.com/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4)

> \[Log Loss\] looks at the probabilities themselves and not just the order of the predictions like AUC.



## Precision-Recall Trade-off

Precision is the percentage of relevant items out of those that have been returned, while recall is the percentage of relevant items that have been returned out of the overall number of relevant items. Hence, it is easy to artificially increase recall to 100% by always returning all the items in the database, but this would mean settling for near-zero precision. Similarly, one can increase precision by always returning a single item that the algorithm is very confident about, but this means that recall would suffer. Ultimately, the best balance between precision and recall depends on the application.

## F1

![{\displaystyle F\_{1}={\frac {2}{{\tfrac {1}{\mathrm {recall} }}+{\tfrac {1}{\mathrm {precision} }}}}=2\cdot {\frac {\mathrm {precision} \cdot \mathrm {recall} }{\mathrm {precision} +\mathrm {recall} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5663ca95d471868169c4e4ea57c936f1b6f4a588)

Often, between the two, there is a trade off. Improvements may increase recall, but lower precision in the process, or vice versa. These are often combined into an F1 score, which is [a type of average that is biased toward the lower of two fractional values](https://en.wikipedia.org/wiki/F1_score). Systems often push for the highest possible F1 score.

An acceptable F1 score depends on the application. There is no absolute number, though an F1 around 80% is typical for a useful system in many applications.

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



[Beyond Accuracy: Precision and Recall – Towards Data Science](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c)

[Unintended Consequences and Goodhart’s Law – Towards Data Science](https://towardsdatascience.com/unintended-consequences-and-goodharts-law-68d60a94705c)

## RMSE

Root Mean Squared Error \(RMSE\)The square root of the mean/average of the square of all of the error.

The use of RMSE is very common and it makes an excellent general purpose error metric for numerical predictions.

Compared to the similar Mean Absolute Error, RMSE amplifies and severely punishes large errors.



## Common visualizations

1. ROC curve
2. Cumulative response curve: tp rate \(tp divided by totally number of positives\) \(y axis\) vs. percentage of the population that is targeted \(x axis\)
3. Lift curve

> One of the reaons accuracy is a poor metric is that it is misleading when daasets are skews…

\(e.g. 93% negatives and 7% positives.\) AUC is a better metric.

> Even modest AUC scores may lead to good business results.

> A critical part of the data scientist's job is aranging for proper evaluation of models and conveying this information to stakeholders. Doing this well takes expereince, but it is vital in order to reduce superises and to manage expectations among all concerned. Visualizatino of reults is an important piece of the evaluation task.
>
> When building a model from data, adjusting the training samplein various ways may be useful or even necessary; but evluation should use a sample reflecting the original, realistic population so that the resutls reflect waht will actually be achieved.
>
> When the costs and benefits of decisions can be specified, the data scientist can calculate an expected cost per instance for each modeland simply choose whicever model produces the best value. In some cases a basic profit graph can be useful to compare modesl of interest under a range of conditions. These graphs may be easy to comprehend for stakeholders who are not data scientists, since they reduce model performance to their basic "bottom line" cost or profit.
>
> The disavantage of a profit graph is that it requires that operating conditions be known and specified exactly. With many real-world problems, the operating conditions are imprecise or change over time, and the data scientist must contend with uncertainty. In such cases other graphs may be more useful. When costs and benefits cannot be specified with confidence, but the class mix will likely not change, a _cumulative response_ or _lift_ graph is useful. Both show the relative advantages of classifiers, independent of the value \(monetary or otherwise\) of the advantages.
>
> Finally, ROC curves are a valuable visualization tool for the data scientist. Though they take some practive to interpret readily, they seperate out performance from operating conditions. In doing so they convey the fundametal trade-offs that each model is making.



## F1

> My identifier has a really great F1. This is the best of both worlds. Both my false positive and false negative rates are low, which means that I can identify POI's reliable and accurately. If my identifier finds a POI then the person is almost certainly a POI, and if the identifier does not flag someone, then they are almost certainly not a POI.

## Precision and Recall

> My identifier doesn't have great **precision**, but it does have good **recall**. That means that, nearly every time a POI shows up in my test set, I am able to identify him or her. The cost of this is that I sometime get some false positives, where non-POIs get flagged.
>
> My identifier doesn't have great **recall**, but it does have good **precision**. That means that whenever a POI gets flagged in my test set, I know with a lot of confidence that it's very likely to be a real POI and not a false alarm. On the other hand, the price I pay for this is that I sometimes miss real POIs, since I'm effectively reluctant to pull the trigger on edge cases.

