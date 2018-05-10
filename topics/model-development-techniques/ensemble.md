# Ensemble

### 

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

[A Kaggler’s Guide to Model Stacking in Practice \| No Free Hunch](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)

