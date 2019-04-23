# Tree

* [kjw0612/awesome-random-forest: Random Forest - a curated list of resources regarding random forest](https://github.com/kjw0612/awesome-random-forest)
* [Introduction to Decision Tree Learning – Heartbeat](https://heartbeat.fritz.ai/introduction-to-decision-tree-learning-cd604f85e236)
* [Quick Guide to Boosting Algorithms in Machine Learning](https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/)
* [Introduction to Random Forest - All About Analytics](https://analyticsdefined.com/introduction-random-forests/)
* [Improving the Random Forest in Python Part 1 – Towards Data Science](https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd)
* [Random Forest Simple Explanation – William Koehrsen – Medium](https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d)
* [Random Forest in Python – Towards Data Science](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0)
* [Improving the Random Forest in Python Part 1 – Towards Data Science](https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd)
* [Hyperparameter Tuning the Random Forest in Python – Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
* [Explaining model's predictions \| Kaggle](https://www.kaggle.com/alijs1/explaining-model-s-predictions/notebook)
* [shubh on Twitter: "In my experiments in almost all cases Light GBM seems to perform better than Xgboost.And some how catboost seems to be slowest and least optimized .Looking for other view points .\#MachineLearning \#Boosting"](https://twitter.com/shub777/status/1014620611761467392)
* [lancifollia/tinygbt: A Tiny, Pure Python implementation of Gradient Boosted Trees.](https://github.com/lancifollia/tinygbt)
* [How decision trees work](https://brohrer.github.io/how_decision_trees_work.html)
* [Decision trees and random forests](https://www.johnwmillr.com/decision-trees-and-random-forests/)
* [Random forest interpretation with scikit-learn \| Diving into data](http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/)
* [Gradient Boosting Explained \| GormAnalysis](https://gormanalysis.com/gradient-boosting-explained/)
* [Complete Guide to Parameter Tuning in Gradient Boosting \(GBM\) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
* [A Practical Guide to Tree Based Learning Algorithms \| Sadanand's Notes](https://sadanand-singh.github.io/posts/treebasedmodels/)
* [隨機森林概述](https://mp.weixin.qq.com/s/mrQRNFovcHFL5NRVwbL2Lw)
* [Adaboost入门教程——最通俗易懂的原理介绍](https://mp.weixin.qq.com/s/wKMqCeGa6SfO4JQ435CSkw)
* [Machine Learning Crash Course: Part 5 - Decision Trees and Ensemble Models](https://ml.berkeley.edu/blog/2017/12/26/tutorial-5/)

A random forest is a bunch of independent decision trees each contributing a “vote” to an prediction. E.g. if there are 50 trees, and 32 say “rainy” and 18 say “sunny”, then the score for “rainy” is 32/50, or 64,% and the score for a “sunny” is 18/50, or 36%. Since 64% &gt; 36%, the forest has voted that they think it will rain.  
  
When you add more decision trees to a random forest, they decide what they think INDEPENDENTLY of all the other trees. They learn on their own, and when it comes time to make a prediction, they all just throw their own uninfluenced opinion into the pot.  
  
A gradient boosting model is a CHAIN of decision trees that also each make a vote. But instead of each learning in isolation, when you add a new one to the chain, it tries to improve a bit on what the rest of the chain already thinks. So, a new tree’s decision IS influenced by all the trees that have already voiced an opinion.  
  
Unlike a Random Forest, when you add a new tree to a GBM, it gets to see what its predecessors thought - and how they got it right or wrong. They then formulate a suggestion to correct the errors of their predecessors - and then they add that to the pot, and then the process continues with the next tree you add to the chain.



## Random Forest

According to [Jeremy](https://youtu.be/CzdWqFTmn0Y?t=36m37s), random forest is a universal machine learning technique.

* It can predict something that can be of any kind — it could be a category \(classification\), a continuous variable \(regression\).
* It can predict with columns of any kind — pixels, zip codes, revenues, etc \(i.e. both structured and unstructured data\).
* It does not generally overfit too badly, and it is very easy to stop it from overfitting.
* You do not need a separate validation set in general. It can tell you how well it generalizes even if you only have one dataset.
* It has few, if any, statistical assumptions. It does not assume that your data is normally distributed, the relationship is linear, or you have specified interactions.
* It requires very few pieces of feature engineering. For many different types of situation, you do not have to take the log of the data or multiply interactions together.

According to [Jeremy](https://youtu.be/CzdWqFTmn0Y?t=4310), random forest works fine with IDs \(i.e. no need to drop them\).

[Jeremy said ](https://youtu.be/blyXCk4sgEg?t=4114)in practice he trains model with 20-30 trees and 1000 trees end of the project or overnight. There is no harm to have more tree except it takes more time to compute.

Jeremy found that 1, 3, 5, 10, 25 are good possible values of `min_samples_leaf`. Hundreds or thousands for really big data set.

The good possible values of `max_features` [Jeremy found ](https://youtu.be/blyXCk4sgEg?t=5173)are 1, .5, log2 or square root.

[Jeremy recommends](https://youtu.be/YSFG_W8JxBo?t=28m48s) that for large data set, use `set_rf_samples(1_000_000)` \(a function in the fast.ai library\) so that each tree is only build using samples of training set. Otherwise it would be too slow.

If you are going to build a number random forest models \(e.g. to tune the hyper-parameters\), run `%time x = np.array(trn, dtype=np.float32)` first as the random forest algorithm in sk-learn will do so anyway \(so sk-learn doesn't need to do so repeatedly\). [Jeremy found this ](https://youtu.be/YSFG_W8JxBo?t=1886)by using `%prun m.fit(x, y)`.

 [Jeremy said](https://youtu.be/0v93qHDqq_g?t=381):

> By decreasing the `set_rf_samples` number, we are actually decreasing the power of the estimator and increasing the correlation...

Why correlation increases? Because high chance that 2 estimators are trained using the same sample?

[Jeremy said ](https://youtu.be/0v93qHDqq_g?t=1m51s)the keys hyper-parameters to play with are `set_rf_samples` \(from the fast.ai library\), `min_samples_leaf` and `max_features`.

### TreeInterpreter

A tool to see, for a given tree-based model, how each feature affect the prediction. [Jeremy said ](https://youtu.be/3jl2h9hSRvc?t=38m2s)it is not useful for Kaggle competition but good in the business world to explain the model.

### OOB Score vs Validation Score

[Jeremy said](https://youtu.be/3jl2h9hSRvc?t=49m23s):

> Actually, in this case, the difference between the OOB \(0.89420\) and the validation \(0.89319\) is actually pretty close. So if there was a big difference, I’d be very worried about whether we’ve dealt with the temporal side of things correctly.

Not quite sure what it means, but I guess he is talking about the case where the validation score is much lower than the OOB score.

### Extrapolation

From description of the [YouTube video](https://www.youtube.com/watch?v=3jl2h9hSRvc):

> Next up, we look into the subtle but important issue of extrapolation. This is the weak point of random forests - they can't predict values outside the range of the input data. We study ways to identify when this problem happens, and how to deal with it.

A good explanation from the comment of the video:

> TLDR: we split data into training set and validation set, where all records in validation set are in future relative to records in training set, so the only way we can predict if something is in validation set is if we have some features which are time dependent, like SaleDate, ModelId etc.
>
> Here is my understanding: First we split data into training set and validation set, where all records in validation set are in future relative to records in training set. Then we added field 'is\_valid' and set it to 1 \(true\) for records that belong to validation set, and to 0 \(false\) for records that are in training set. Then we create a model that tries to predict whether particular record belongs to training set or validation set \(just like previously we tried to predict SalePrice, so 'is\_valid' is our new dependent variable\). As we spit data into two sets based solely on record dates \(see \#1\), then the only way we can predict if record belongs to training or validations set is if we have any other fields which are time dependent. For example, if all records in validation set have SaleDate &gt;= 1 Jan 2018, and records in training set have SaleDate &lt; 1 Jan 2018, and SaleDate is one of the features we can look at, then random forest can use 'SaleDate &gt;= 1 Jan 2018' to predict 'is\_valid" value. Another less obvious example might be something like ModelId, which can be an autoincrement field in DB and all values in training set will have ModelId &lt; 420000, and values in validation set have ModelId &gt;= 42000 simply because as somebody added new Models into DB they got higher ModelId due to autoincrement. Such features will have high feature importance for predicting 'is\_valid'.

[Jeremy suggests ](https://youtu.be/3jl2h9hSRvc?t=49m23s)to try to drop the temporal variables. It is done by first to change the target variable to `is_valid` whether the sample is in validation set and train the model to see if any of the features is strong at predicting this. Then try to drop them one by one and train the original model to see if the score gets better.

### Final Model

Here's how Jeremy train the final model \(after feature engineering, feature selection and hyper-parameter tuning\)

* Run `reset_rf_samples()` so that each estimator use the full training set.
* Use large number of estimator, e.g. `reset_rf_samples()n_estimators=160`.

### Random Forest vs GBM

[Jeremy said ](https://youtu.be/O5F9vR2CNYI?t=328)random forests have a nice benefit over GBMs that they are harder to screw up and easier to scale.

## Gradient boosting

Gradient boosting is a type of boosting. It relies on the intuition that the best possible next model, when combined with previous models, minimizes the overall prediction error. The key idea is to set the target outcomes for this next model in order to minimize the error. How are the targets calculated? The target outcome for each case in the data depends on how much changing that case’s prediction impacts the overall prediction error:  
  
If a small change in the prediction for a case causes a large drop in error, then next target outcome of the case is a high value. Predictions from the new model that are close to its targets will reduce the error.  
If a small change in the prediction for a case causes no change in error, then next target outcome of the case is zero. Changing this prediction does not decrease the error.  
The name gradient boosting arises because target outcomes for each case are set based on the gradient of the error with respect to the prediction. Each new model takes a step in the direction that minimizes prediction error, in the space of possible predictions for each training case.

[How to explain gradient boosting](http://explained.ai/gradient-boosting/index.html)



## Ensembles and boosting

Machine learning models can be fitted to data individually, or combined in an ensemble. An ensemble is a combination of simple individual models that together create a more powerful new model.  
  
Boosting is a method for creating an ensemble. It starts by fitting an initial model \(e.g. a tree or linear regression\) to the data. Then a second model is built that focuses on accurately predicting the cases where the first model performs poorly. The combination of these two models is expected to be better than either model alone. Repeat the process many times. Each successive model attempts to correct for the shortcomings of the combined boosted ensemble of all previous models.

