# Model Development Techniques

[How to unit test machine learning code. – Chase Roberts – Medium](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765)

[Top 6 errors novice machine learning engineers make](https://medium.com/ai³-theory-practice-business/top-6-errors-novice-machine-learning-engineers-make-e82273d394db)

### Hyperparameter tuning

* [Bayesian Optimization for Hyperparameter Tuning](https://arimo.com/data-science/2016/bayesian-optimization-hyperparameter-tuning/)
* [Hyperparameter Tuning with hyperopt in Python](http://steventhornton.ca/hyperparameter-tuning-with-hyperopt-in-python/)
* [Best Practices for Parameter Tuning on Models](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion/19083)
* [Avoiding Grid for parameter tuning](https://www.kaggle.com/c/allstate-claims-severity/discussion/24532)
* [Bayesian Optimization XGBoost parameters](https://www.kaggle.com/tilii7/bayesian-optimization-xgboost-parameters)
* [XGboost + Bayesian Optimization](https://www.kaggle.com/tilii7/xgboost-bayesian-optimization/code)
* [SVR+sparse matrix+Bayesian optimization](https://www.kaggle.com/tilii7/svr-sparse-matrix-bayesian-optimization/)
* [Bayesian Optimization of a Technical Trading Algorithm with Zipline+SigOpt](https://blog.quantopian.com/bayesian-optimization-of-a-technical-trading-algorithm-with-ziplinesigopt-2/)
* [sklearn-gridsearchcv-replacement.ipynb](https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/sklearn-gridsearchcv-replacement.ipynb)
* [BayesianOptimization/sklearn\_example.py](https://github.com/fmfn/BayesianOptimization/blob/master/examples/sklearn_example.py)
* [Hyper-parameter Optimization with keras?](https://github.com/fchollet/keras/issues/1591)
* [Keras + Hyperopt: A very simple wrapper for convenient hyperparameter optimization](https://github.com/maxpumperla/hyperas)
* [Effectively running thousands of experiments: Hyperopt with Sacred](https://gab41.lab41.org/effectively-running-thousands-of-experiments-hyperopt-with-sacred-dfa53b50f1ec)
* [Hyperopt tutorial for Optimizing Neural Networks' Hyperparameters - Vooban](https://vooban.com/en/tips-articles-geek-stuff/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters/)
* [hyperopt\_experiments.py](https://github.com/Lab41/pythia/blob/master/experiments/hyperopt_experiments.py)
* [Spearmint](https://github.com/HIPS/Spearmint)
* [GPyOpt](https://sheffieldml.github.io/GPyOpt/) a Python open-source library for Bayesian Optimization
* [Scikit-Optimize](https://scikit-optimize.github.io/)
* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-3/#hyper)

### Cross-validation

* [Example of CV](https://www.kaggle.com/rspadim/off-example-of-cv/)
* [Cross- Validation Code Visualization: Kind of Fun](https://medium.com/towards-data-science/cross-validation-code-visualization-kind-of-fun-b9741baea1f8)

### Ensemble

* [Ensemble Learning to Improve Machine Learning Results](https://blog.statsbot.co/ensemble-learning-d1dcd548e936)
* [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)

> Common approaches of ensemble learning are:
>
> * **Bagging**: Use different random subsets of training data to train each base model. Then all the base models vote to generate the final predictions. This is how random forest works.
> * **Boosting**: Train base models iteratively, modify the weights of training samples according to the last iteration. This is how gradient boosted trees work. \(Actually it’s not the whole story. Apart from boosting, GBTs try to learn the residuals of earlier iterations.\) It performs better than bagging but is more prone to overfitting.
> * **Blending**: Use non-overlapping data to train different base models and take a weighted average of them to obtain the final predictions. This is easy to implement but uses less data.
> * **Stacking**: To be discussed next.
>
> In theory, for the ensemble to perform well, two factors matter:
>
> * **Base models should be as unrelated as possibly**. This is why we tend to include non-tree-based models in the ensemble even though they don’t perform as well. The math says that the greater the diversity, and less bias in the final ensemble.
> * **Performance of base models shouldn’t differ to much.**

> Compared with blending, stacking makes better use of training data. Here’s a diagram of how it works:
>
> ![](http://7xlo8f.com1.z0.glb.clouddn.com/blog-diagram-stacking.jpg)
>
> It’s much like cross validation. Take 5-fold stacking as an example. First we split the training data into 5 folds. Next we will do 5 iterations. In each iteration, train every base model on 4 folds and predict on the hold-out fold. **You have to keep the predictions on the testing data as well.** This way, in each iteration every base model will make predictions on 1 fold of the training data and all of the testing data. After 5 iterations we will obtain a matrix of shape `#(samples in training data) X #(base models)`. This matrix is then fed to the stacker \(it’s just another model\) in the second level. After the stacker is fitted, use the predictions on testing data by base models \(**each base model is trained 5 times, therefore we have to take an average to obtain a matrix of the same shape**\) as the input for the stacker and obtain our final predictions.
>
> Maybe it’s better to just show the codes:

```python
class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
```

> Prize winners usually have larger and much more complicated ensembles. For beginner, implementing a correct 5-fold stacking is good enough.

[Introduction to Python Ensembles](https://www.dataquest.io/blog/introduction-to-ensembles/)

[Kaggle Ensembling Guide \| MLWave](https://mlwave.com/kaggle-ensembling-guide/)

[Introduction to Ensembling/Stacking in Python \| Kaggle](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook)

