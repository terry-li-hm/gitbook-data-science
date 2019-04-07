# XGBoost

* [**Complete Guide to Parameter Tuning in XGBoost \(with codes in Python\)**](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
* [**A Gentle Introduction to XGBoost for Applied Machine Learning**](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)
* [Interpretable Machine Learning with XGBoost – Towards Data Science](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27)



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



> **XGBoost** is the leading model for working with standard tabular data \(the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos\). XGBoost models dominate many Kaggle competitions.
>
> To reach peak accuracy, XGBoost models require more knowledge and _model tuning_ than techniques like Random Forest. After this tutorial, you'ill be able to
>
> * Follow the full modeling workflow with XGBoost
> * Fine-tune XGBoost models for optimal performance
>
> XGBoost is an implementation of the **Gradient Boosted Decision Trees** algorithm \(scikit-learn has another version of this algorithm, but XGBoost has some technical advantages.\) What is **Gradient Boosted Decision Trees**? We'll walk through a diagram.
>
> ![xgboost image](https://i.imgur.com/e7MIgXk.png)
>
> We go through cycles that repeatedly builds new models and combines them into an **ensemble** model. We start the cycle by calculating the errors for each observation in the dataset. We then build a new model to predict those. We add predictions from this error-predicting model to the "ensemble of models."
>
> To make a prediction, we add the predictions from all previous models. We can use these predictions to calculate new errors, build the next model, and add it to the ensemble.
>
> There's one piece outside that cycle. We need some base prediction to start the cycle. In practice, the initial predictions can be pretty naive. Even if it's predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.

> XGBoost has a few parameters that can dramatically affect your model's accuracy and training speed. The first parameters you should understand are:
>
> #### n\_estimators and early\_stopping\_rounds <a id="n_estimators-and-early_stopping_rounds"></a>
>
> **n\_estimators** specifies how many times to go through the modeling cycle described above.
>
> In the [underfitting vs overfitting graph](http://i.imgur.com/2q85n9s.png), n\_estimators moves you further to the right. Too low a value causes underfitting, which is inaccurate predictions on both training data and new data. Too large a value causes overfitting, which is accurate predictions on training data, but inaccurate predictions on new data \(which is what we care about\). You can experiment with your dataset to find the ideal. Typical values range from 100-1000, though this depends a lot on the **learning rate**discussed below.
>
> The argument **early\_stopping\_rounds** offers a way to automatically find the ideal value. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n\_estimators. It's smart to set a high value for **n\_estimators** and then use **early\_stopping\_rounds** to find the optimal time to stop iterating.
>
> Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping. **early\_stopping\_rounds = 5** is a reasonable value. Thus we stop after 5 straight rounds of deteriorating validation scores.
>
> When using **early\_stopping\_rounds**, you need to set aside some of your data for checking the number of rounds to use. If you later want to fit a model with all of your data, set **n\_estimators** to whatever value you found to be optimal when run with early stopping.
>
> #### learning\_rate <a id="learning_rate"></a>
>
> Here's a subtle but important trick for better XGBoost models:
>
> Instead of getting predictions by simply adding up the predictions from each component model, we will multiply the predictions from each model by a small number before adding them in. This means each tree we add to the ensemble helps us less. In practice, this reduces the model's propensity to overfit.
>
> So, you can use a higher value of **n\_estimators** without overfitting. If you use early stopping, the appropriate number of trees will be set automatically.
>
> In general, a small learning rate \(and large number of estimators\) will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle.
>
> #### n\_jobs <a id="n_jobs"></a>
>
> On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter **n\_jobs** equal to the number of cores on your machine. On smaller datasets, this won't help.
>
> The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. But, it's useful in large datasets where you would otherwise spend a long time waiting during the `fit` command.
>
> XGBoost has a multitude of other parameters, but these will go a very long way in helping you fine-tune your XGBoost model for optimal performance.

* [Unveiling Mathematics behind XGBoost – Ajit Samudrala – Medium](https://medium.com/@samudralaajit/unveiling-mathematics-behind-xgboost-c7f1b8201e2a)
* [Fine-tuning XGBoost in Python like a boss – Towards Data Science](https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e)
* \*\*\*\*[XGBoost 入门系列第一讲](https://bigquant.com/community/t/topic/36)

