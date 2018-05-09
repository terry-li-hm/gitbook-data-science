# Anomaly Detection

* 对于非平衡样本问题，用PR曲线能有效的看出模型性能。



[What metrics should be used for evaluating a model on an imbalanced data set?](https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba)

[Introduction to Anomaly Detection: Concepts and Techniques \| My views of the World and Systems](https://iwringer.wordpress.com/2015/11/17/anomaly-detection-concepts-and-techniques/)

[anomalize: Tidy Anomaly Detection](http://www.business-science.io/code-tools/2018/04/08/introducing-anomalize.html)

[Machine learning best practices: detecting rare events - Subconscious Musings](https://blogs.sas.com/content/subconsciousmusings/2017/07/19/machine-learning-best-practices-detecting-rare-events/)

[Does Balancing Classes Improve Classifier Performance? – Win-Vector Blog](http://www.win-vector.com/blog/2015/02/does-balancing-classes-improve-classifier-performance/)

[Credit Card Fraud Detection using Autoencoders in Keras — TensorFlow for Hackers \(Part VII\)](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd)

[Which machine learning techniques have you used for fraud detection, and why? Do you prefer statistical techniques versus artificial intelligence? - Quora](https://www.quora.com/Which-machine-learning-techniques-have-you-used-for-fraud-detection-and-why-Do-you-prefer-statistical-techniques-versus-artificial-intelligence)

[How PayPal Is Taking a Chance on AI to Fight Fraud \| American Banker](https://www.americanbanker.com/news/how-paypal-is-taking-a-chance-on-ai-to-fight-fraud)

[scikit-learn-contrib/imbalanced-learn: Python module to perform under sampling and over sampling with various techniques.](https://github.com/scikit-learn-contrib/imbalanced-learn)

[How to handle Imbalanced Classification Problems in machine learning?](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/)

> Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of misclassification of the minority class as compared to the majority class.

[8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

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
> * **Kappa \(or **[**Cohen’s kappa**](https://en.wikipedia.org/wiki/Cohen%27s_kappa)**\)**: Classification accuracy normalized by the imbalance of the classes in the data.
> * **ROC Curves**: Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.
>
> You can learn a lot more about using ROC Curves to compare classification accuracy in our post “[Assessing and Comparing Classifier Performance with ROC Curves](http://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/)“.
>
> Still not sure? Start with kappa, it will give you a better idea of what is going on than classification accuracy.

> Consider testing under-sampling when you have an a lot data \(tens- or hundreds of thousands of instances or more\)Consider testing over-sampling when you don’t have a lot of data \(tens of thousands of records or less\)Consider testing random and non-random \(e.g. stratified\) sampling schemes.Consider testing different resampled ratios \(e.g. you don’t have to target a 1:1 ratio in a binary classification problem, try other ratios

[Imbalanced Data Classification \| An Explorer of Things](https://chih-ling-hsu.github.io/2017/07/25/Imbalanced-Data-Classification)

[Dealing with unbalanced data in machine learning](https://shiring.github.io/machine_learning/2017/04/02/unbalanced)

