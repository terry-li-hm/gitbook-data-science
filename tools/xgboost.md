# XGBoost

[Complete Guide to Parameter Tuning in XGBoost \(with codes in Python\)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)



> 1. Reserve a portion of training set as the validation set.
> 2. Set `eta` to a relatively high value \(e.g. 0.05 ~ 0.1\), `num_round` to 300 ~ 500.
> 3. Use grid search to find the best combination of other parameters.
> 4. Gradually lower `eta` until we reach the optimum.
> 5. **Use the validation set as `watch_list` to re-train the model with the best parameters. Observe how score changes on validation set in each iteration. Find the optimal value for `early_stopping_rounds`.**



[Introduction to Boosted Trees — xgboost 0.71 documentation](http://xgboost.readthedocs.io/en/latest/model.html)

> So random forests and boosted trees are not different in terms of model, the difference is how we train them.

> After introducing the model, let us begin with the real training part. How should we learn the trees? The answer is, as is always for all supervised learning models: _define an objective function, and optimize it_!

* XGBoost is an implementation of _boosted tree_.
* _Boosted tree_ is an algorithm to train _tree ensemble_. _Random forests_ is another algorithm to train _tree ensemble_.
* _Tree ensemble_ is a model.
* Model in supervised learning usually refers to the mathematical structure of how to make the prediction yi given xi. For example, a common model is a _linear model_, where the prediction is given by y^i=∑jθjxijy^i=∑jθjxij, a linear combination of weighted input features.
* The tree ensemble model is a set of classification and regression trees \(CART\).
*  A CART is a bit different from decision trees, where the leaf only contains decision values. In CART, a real score is associated with each of the leaves, which gives us richer interpretations that go beyond classification.
* **Gradient boosting** is a [machine learning](https://en.wikipedia.org/wiki/Machine_learning) technique for [regression](https://en.wikipedia.org/wiki/Regression_%28machine_learning%29) and [classification](https://en.wikipedia.org/wiki/Classification_%28machine_learning%29) problems, which produces a prediction model in the form of an [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) of weak prediction models, typically [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning). It builds the model in a stage-wise fashion like other [boosting](https://en.wikipedia.org/wiki/Boosting_%28meta-algorithm%29) methods do, and it generalizes them by allowing optimization of an arbitrary [differentiable](https://en.wikipedia.org/wiki/Differentiable_function) [loss function](https://en.wikipedia.org/wiki/Loss_function). \([Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting)\)
* Boosting is one of several classic methods for creating ensemble models, along with bagging, random forests, and so forth. \([Microsoft](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/boosted-decision-tree-regression)\)
* 说到这里可能有人会问，为什么我没有听过这个名字。这是因为Boosted Tree有各种马甲，比如GBDT, GBRT \(gradient boosted regression tree\)，MART11，LambdaMART也是一种boosted tree的变种。\( [我爱计算机](http://www.52cs.org/)\)
* _GBM_ \(Gradient Boosting Machine\). XGBoost is a better algorithm than GBM.

[Interpretable Machine Learning with XGBoost – Towards Data Science](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27)

