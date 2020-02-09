# Anomaly Detection \(Outlier Detection\)

Opinions

> 無監督學習現階段還是沒有監督學習準確，模型還是靠監督學習算法的，無監督學習可能很多用在風險特徵的發現

[Microsoft](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice):

![](../.gitbook/assets/image%20%2837%29.png)

![](../.gitbook/assets/image%20%2855%29.png)

[How to Successfully Harness AI to Combat Fraud and Abuse - RSA 2018 - YouTube](https://www.youtube.com/watch?v=5gxI-6QmPdE)

## Relevant Kaggle Competitions

* [TalkingData AdTracking Fraud Detection Challenge \| Kaggle](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/kernels)

## Wiki

> Several anomaly detection techniques have been proposed in literature. Some of the popular techniques are:
>
> * Density-based techniques \([k-nearest neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm),[\[6\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-6)[\[7\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-7)[\[8\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-8) [local outlier factor](https://en.wikipedia.org/wiki/Local_outlier_factor),[\[9\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-9) and many more variations of this concept[\[10\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-10)\).
> * Subspace-[\[11\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-11) and correlation-based[\[12\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-12) outlier detection for high-dimensional data.[\[13\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-13)
> * One-class [support vector machines](https://en.wikipedia.org/wiki/Support_vector_machines).[\[14\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-14)
> * Replicator [neural networks](https://en.wikipedia.org/wiki/Neural_network).[\[15\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-replicator-15)
> * [Bayesian Networks](https://en.wikipedia.org/wiki/Bayesian_Network).[\[15\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-replicator-15)
> * [Hidden Markov models](https://en.wikipedia.org/wiki/Hidden_Markov_model) \(HMMs\).[\[15\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-replicator-15)
> * [Cluster analysis](https://en.wikipedia.org/wiki/Cluster_analysis)-based outlier detection.[\[16\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-16)[\[17\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-17)
> * Deviations from [association rules](https://en.wikipedia.org/wiki/Association_rule_learning) and frequent itemsets.
> * Fuzzy logic-based outlier detection.
> * [Ensemble techniques](https://en.wikipedia.org/wiki/Ensemble_learning), using [feature bagging](https://en.wikipedia.org/wiki/Random_subspace_method),[\[18\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-18)[\[19\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-19) score normalization[\[20\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-20)[\[21\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-21) and different sources of diversity.[\[22\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-22)[\[23\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-23)
>
> The performance of different methods depends a lot on the data set and parameters, and methods have little systematic advantages over another when compared across many data sets and parameters.[\[24\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-CamposZimek2016-24)[\[25\]](https://en.wikipedia.org/wiki/Anomaly_detection#cite_note-25)

## Hierarchical Temporal Memory \(HTM\)

* [Hierarchical Temporal Memory for Real-time Anomaly Detection](https://www.dropbox.com/s/zvv0ov63pel9je7/anomalydetectionhtmmeetup-170829200447.pdf?dl=0)
* [Anomaly Detection — NuPIC 1.0.4 documentation](http://nupic.docs.numenta.org/stable/guides/anomaly-detection.html)

## Types of Anomaly

[Page 7 to 10](https://www.dropbox.com/s/zvv0ov63pel9je7/anomalydetectionhtmmeetup-170829200447.pdf?dl=0)

## Benchmark

[numenta/NAB: The Numenta Anomaly Benchmark](https://github.com/numenta/NAB)

## Visualization

[Visualization on a 2D map \(with t-SNE\) \| Kaggle](https://www.kaggle.com/cherzy/visualization-on-a-2d-map-with-t-sne/code)

## Maybe read later

* [Time Series Anomaly Detection Algorithms – Stats and Bots](https://blog.statsbot.co/time-series-anomaly-detection-algorithms-1cef5519aef2)
* [Semi-Supervised Anomaly Detection Survey \| Kaggle](https://www.kaggle.com/matheusfacure/semi-supervised-anomaly-detection-survey/code)
* [Anomaly Detection for Airbnb’s Payment Platform – Airbnb Engineering & Data Science – Medium](https://medium.com/airbnb-engineering/anomaly-detection-for-airbnb-s-payment-platform-e3b0ec513199)
* [Outlier and fraud detection using Hadoop](https://www.slideshare.net/pkghosh99/outlier-and-fraud-detection)
* [Relative Density and Outliers \| Mawazo](https://pkghosh.wordpress.com/2012/10/18/relative-density-and-outliers/)
* [It’s a lonely life for outliers \| Mawazo](https://pkghosh.wordpress.com/2012/06/18/its-a-lonely-life-for-outliers/)
* [Fraudsters, Outliers and Big Data \| Mawazo](https://pkghosh.wordpress.com/2012/01/02/fraudsters-outliers-and-big-data-2/)
* [机器学习-异常检测算法（一）：Isolation Forest](https://zhuanlan.zhihu.com/p/27777266)
* [机器学习-异常检测算法（二）：Local Outlier Factor](https://zhuanlan.zhihu.com/p/28178476)
* [機器學習-異常檢測算法（三）：Principal Component Analysis](https://zhuanlan.zhihu.com/p/29091645)
* [Engineering Extreme Event Forecasting at Uber with Recurrent Neural Networks \| Uber Engineering Blog](https://eng.uber.com/neural-networks/)
* [Identifying Outages with Argos - Uber Engineering Blog](https://eng.uber.com/argos/)
* [A Brief Overview of Outlier Detection Techniques – Towards Data Science](https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561)
* [3 methods to deal with outliers \| Neural Designer](https://www.neuraldesigner.com/blog/3_methods_to_deal_with_outliers)
* [How to Identify Outliers in your Data](https://machinelearningmastery.com/how-to-identify-outliers-in-your-data/)
* [Introduction to Anomaly Detection](https://www.datascience.com/blog/python-anomaly-detection)
* [How PayPal Is Taking a Chance on AI to Fight Fraud \| American Banker](https://www.americanbanker.com/news/how-paypal-is-taking-a-chance-on-ai-to-fight-fraud)
* [Which machine learning techniques have you used for fraud detection, and why? Do you prefer statistical techniques versus artificial intelligence? - Quora](https://www.quora.com/Which-machine-learning-techniques-have-you-used-for-fraud-detection-and-why-Do-you-prefer-statistical-techniques-versus-artificial-intelligence)
* [dalpozz/AMLFD: Adaptive Machine Learning for Credit Card Fraud Detection](https://github.com/dalpozz/AMLFD)
* [Adaptive Machine Learning for Credit Card Fraud Detection](http://www.ulb.ac.be/di/map/adalpozz/pdf/Dalpozzolo2015PhD.pdf)
* [Credit Card Fraud Detection using Autoencoders in Keras — TensorFlow for Hackers \(Part VII\)](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd)
* [Fraud deep learning\_v2](https://www.slideshare.net/RatnakarPandey6/fraud-deep-learningv2)

## Metric

* [What metrics should be used for evaluating a model on an imbalanced data set?](https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba)
* [What are the best performance measures for an anomaly detection model? - Quora](https://www.quora.com/What-are-the-best-performance-measures-for-an-anomaly-detection-model)

> 对于非平衡样本问题，用PR曲线能有效的看出模型性能。

## Feature Engineering

> a fraud detection model may use anomaly detection scores as an extra generated feature going into the supervised learning algorithm.

## Algorithms

> Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of misclassification of the minority class as compared to the majority class.

## Imbalanced Classes

* [scikit-learn-contrib/imbalanced-learn: Python module to perform under sampling and over sampling with various techniques.](https://github.com/scikit-learn-contrib/imbalanced-learn)
* [How to Handle Imbalanced Classes in Machine Learning](https://elitedatascience.com/imbalanced-classes)
* [Dealing with imbalanced data: undersampling, oversampling and proper cross-validation](https://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation)
* [Unbalanced data and cross-validation](https://www.kaggle.com/questions-and-answers/27589)
* [How To handle Imbalance Data : Study in Detail \| Kaggle](https://www.kaggle.com/gargmanish/how-to-handle-imbalance-data-study-in-detail)
* [Does Balancing Classes Improve Classifier Performance? – Win-Vector Blog](http://www.win-vector.com/blog/2015/02/does-balancing-classes-improve-classifier-performance/)
* [Machine learning best practices: detecting rare events - Subconscious Musings](https://blogs.sas.com/content/subconsciousmusings/2017/07/19/machine-learning-best-practices-detecting-rare-events/)
* [Imbalanced Data Classification \| An Explorer of Things](https://chih-ling-hsu.github.io/2017/07/25/Imbalanced-Data-Classification)
* [Dealing with unbalanced data in machine learning](https://shiring.github.io/machine_learning/2017/04/02/unbalanced)
* [How to handle Imbalanced Classification Problems in machine learning?](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/)
* [Training artificial intelligence with artificial X-rays: New research could help AI identify rare conditions in medical images by augmenting existing datasets -- ScienceDaily](https://www.sciencedaily.com/releases/2018/07/180706150816.htm)
* [\[1710.05381\] A systematic study of the class imbalance problem in convolutional neural networks](https://arxiv.org/abs/1710.05381)
* [8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

> Consider testing under-sampling when you have an a lot data \(tens- or hundreds of thousands of instances or more\)Consider testing over-sampling when you don’t have a lot of data \(tens of thousands of records or less\)Consider testing random and non-random \(e.g. stratified\) sampling schemes.Consider testing different resampled ratios \(e.g. you don’t have to target a 1:1 ratio in a binary classification problem, try other ratios

> Decompose your larger class into smaller number of other classes…  
>   
> …use a One Class Classifier… \(e.g. treat like outlier detection\)  
>   
> …resampling the unbalanced training set into not one balanced set, but several. Running an ensemble of classifiers on these sets could produce a much better result than one classifier alone

> 6\) Try Penalized Models  
>   
> You can use the same algorithms but give them a different perspective on the problem.  
>   
> Penalized classification imposes an additional cost on the model for making classification mistakes on the minority class during training. These penalties can bias the model to pay more attention to the minority class.  
>   
> Often the handling of class penalties or weights are specialized to the learning algorithm. There are penalized versions of algorithms such as penalized-SVM and penalized-LDA.  
>   
> It is also possible to have generic frameworks for penalized models. For example, Weka has a CostSensitiveClassifier that can wrap any classifier and apply a custom penalty matrix for miss classification.  
>   
> Using penalization is desirable if you are locked into a specific algorithm and are unable to resample or you’re getting poor results. It provides yet another way to “balance” the classes. Setting up the penalty matrix can be complex. You will very likely have to try a variety of penalty schemes and see what works best for your problem.

> A simple way to generate synthetic samples is to randomly sample the attributes from instances in the minority class.  
>   
> You could sample them empirically within your dataset or you could use a method like Naive Bayes that can sample each attribute independently when run in reverse. You will have more and different data, but the non-linear relationships between the attributes may not be preserved.  
>   
> There are systematic algorithms that you can use to generate synthetic samples. The most popular of such algorithms is called SMOTE or the Synthetic Minority Over-sampling Technique.  
>   
> As its name suggests, SMOTE is an oversampling method. It works by creating synthetic samples from the minor class instead of creating copies. The algorithm selects two or more similar instances \(using a distance measure\) and perturbing an instance one attribute at a time by a random amount within the difference to the neighboring instances.  
>   
> Learn more about SMOTE, see the original 2002 paper titled “SMOTE: Synthetic Minority Over-sampling Technique“.  
>   
> There are a number of implementations of the SMOTE algorithm, for example:  
>   
> In Python, take a look at the “UnbalancedDataset” module. It provides a number of implementations of SMOTE as well as various other resampling techniques that you could try.

> I would also advice you to take a look at the following:
>
> * **Kappa \(or** [**Cohen’s kappa**](https://en.wikipedia.org/wiki/Cohen%27s_kappa)**\)**: Classification accuracy normalized by the imbalance of the classes in the data.
> * **ROC Curves**: Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.
>
> You can learn a lot more about using ROC Curves to compare classification accuracy in our post “[Assessing and Comparing Classifier Performance with ROC Curves](http://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/)“.
>
> Still not sure? Start with kappa, it will give you a better idea of what is going on than classification accuracy.

### Negative Mining

> **Negative mining.** The third group of sampling methods is a bit more complex but indeed the most powerful one. Instead of over- or undersampling, we choose the samples intentionally. Although we have much more samples of the frequent class we care most about the most difficult samples, i.e. the samples which are misclassified with the highest probabilities. Thus, we can regularly evaluate the model during training and investigate the samples to identify those that are misclassified more likely. This enables us to wisely select the samples that are shown to the algorithm more often.

![](../.gitbook/assets/image%20%2821%29.png)



