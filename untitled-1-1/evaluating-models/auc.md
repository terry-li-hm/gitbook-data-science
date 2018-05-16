# AUC

[ROC curves and Area Under the Curve explained \(video\)](http://www.dataschool.io/roc-curves-and-auc-explained/)

> \[AUC\] is unaffected by scaling and similar, so it is a good metric for testing the predictive power of individual features.



[anokas](https://www.kaggle.com/anokas/talkingdata-adtracking-eda/notebook):

> Looking at the evaluation page, we can see that the evaluation metric used is **ROC-AUC** \(the area under a curve on a Receiver Operator Characteristic graph\). In english, this means a few important things:
>
> * This competition is a **binary classification** problem - i.e. our target variable is a binary attribute \(Is the user making the click fraudlent or not?\) and our goal is to classify users into "fraudlent" or "not fraudlent" as well as possible
> * Unlike metrics such as [LogLoss](http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/), the AUC score only depends on **how well you well you can separate the two classes**. In practice, this means that only the order of your predictions matter,
>   * As a result of this, any rescaling done to your model's output probabilities will have no effect on your score. In some other competitions, adding a constant or multiplier to your predictions to rescale it to the distribution can help but that doesn't apply here.
>
> If you want a more intuitive explanation of how AUC works, I recommend [this post](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it).

