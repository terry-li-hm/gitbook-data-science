---
description: >-
  Coming up with features is difficult, time-consuming, requires expert
  knowledge. "Applied machine learning" is basically feature engineering.
  — Andrew Ng
---

# Feature Engineering

* [Featuretools/featuretools: automated feature engineering](https://github.com/Featuretools/featuretools)
* [Best Practices for Feature Engineering](https://elitedatascience.com/feature-engineering-best-practices)
* [Feature Engineering 相關文章推薦](https://medium.com/@drumrick/feature-engineering-%E7%9B%B8%E9%97%9C%E6%96%87%E7%AB%A0%E6%8E%A8%E8%96%A6-b4c2aaffe93d)
* [Feature Engineering 特徵工程中常見的方法](https://vinta.ws/code/feature-engineering.html)
* [Feature Engineering - Handling Cyclical Features](http://blog.davidkaleko.com/feature-engineering-cyclical-features.html)
* [使用sklearn做單機特徵工程 - jasonfreak - 博客園](http://www.cnblogs.com/jasonfreak/p/5448385.html)
* [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
* [机器学习中，有哪些特征选择的工程方法？ - 知乎](https://www.zhihu.com/question/28641663)
* [target encoding for categorical features](https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features)
* [Python target encoding for categorical features \| Kaggle](https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features)
* [如何有效處理特徵範圍差異大且類型不一的數據？ - 知乎](https://www.zhihu.com/question/42341851/answer/207949804)
* [An Introduction to Deep Learning for Tabular Data · fast.ai](http://www.fast.ai/2018/04/29/categorical-embeddings/)
* [How to deal with Features having high cardinality](https://www.kaggle.com/general/16927)
* [Transform anything into a vector – Insight Data](https://blog.insightdatascience.com/entity2vec-dad368c5b830)
* [Dimensionality Reduction Algorithms: Strengths and Weaknesses](https://elitedatascience.com/dimensionality-reduction-algorithms)
* [Three Effective Feature Selection Strategies – AI³ \| Theory, Practice, Business – Medium](https://medium.com/ai³-theory-practice-business/three-effective-feature-selection-strategies-e1f86f331fb1)
* [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/)
* [Feature Engineering for Machine Learning \| Udemy](https://www.udemy.com/feature-engineering-for-machine-learning/)
* [Open Machine Learning Course. Topic 6. Feature Engineering and Feature Selection](https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-6-feature-engineering-and-feature-selection-8b94f870706a)
* [A Complete Machine Learning Walk-Through in Python: Part One](https://towardsdatascience.com/a-complete-machine-learning-walk-through-in-python-part-one-c62152f39420)
* [Automated Feature Engineering in Python – Towards Data Science](https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219)
* [A Feature Selection Tool for Machine Learning in Python](https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0)
* [數據科學中的陷阱：定性變量的處理 \| 機器之心](https://www.jiqizhixin.com/articles/2018-07-09-19)
* [plasticityai/magnitude: A fast, efficient universal vector embedding utility package.](https://github.com/plasticityai/magnitude)
* [featuretools-workshop/featuretools-workshop.ipynb at master · fred-navruzov/featuretools-workshop](https://github.com/fred-navruzov/featuretools-workshop/blob/master/featuretools-workshop.ipynb)
* [Featuretools for Good \| Kaggle](https://www.kaggle.com/willkoehrsen/featuretools-for-good)
* [Feature Engineering 相關文章推薦 – Rick Liu – Medium](https://medium.com/@drumrick/feature-engineering-相關文章推薦-b4c2aaffe93d)
* [Why Automated Feature Engineering Will Change the Way You Do Machine Learning](https://towardsdatascience.com/why-automated-feature-engineering-will-change-the-way-you-do-machine-learning-5c15bf188b96)
* [Deep Feature Synthesis: How Automated Feature Engineering Works \| Feature Labs](https://www.featurelabs.com/blog/deep-feature-synthesis/)



## Useful Approaches

* [Aggregating Numeric Columns](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering#Aggregating-Numeric-Columns)
* [Aggregating Categorical Columns](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering#Categorical-Variables)



 

## Automated

* [A Hands on Guide to Automated Feature Engineering using Featuretools](https://www.analyticsvidhya.com/blog/2018/08/guide-automated-feature-engineering-featuretools-python/)
* [automated-feature-engineering/Automated\_Feature\_Engineering.ipynb at master · WillKoehrsen/automated-feature-engineering](https://github.com/WillKoehrsen/automated-feature-engineering/blob/master/walk_through/Automated_Feature_Engineering.ipynb)
* [如何用Python做自动化特征工程 \| 机器之心](https://www.jiqizhixin.com/articles/2018-09-03-4)



## Scaling

[scikit-learn](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py):

> Indeed many estimators are designed with the assumption that each feature takes values close to zero or more importantly that all features vary on comparable scales. In particular, metric-based and gradient-based estimators often assume approximately standardized data \(centered features with unit variances\). A notable exception are decision tree-based estimators that are robust to arbitrary scaling of the data.









[machinelearningmastery.com](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/):

#### Decompose Categorical Attributes

Imagine you have a categorical attribute, like “_Item\_Color_” that can be _Red_, _Blue_ or _Unknown_.

_Unknown_ may be special, but to a model, it looks like just another colour choice. It might be beneficial to better expose this information.

You could create a new binary feature called “_Has\_Color_” and assign it a value of “_1_” when an item has a color and “_0_” when the color is unknown.

Going a step further, you could create a binary feature for each value that _Item\_Color_ has. This would be three binary attributes: _Is\_Red_, _Is\_Blue_ and _Is\_Unknown_.

These additional features could be used instead of the _Item\_Color_ feature \(if you wanted to try a simpler linear model\) or in addition to it \(if you wanted to get more out of something like a decision tree\).

