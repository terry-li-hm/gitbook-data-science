# Inbox





#### a introduction of Naive Bayes Theorem

> Bayes theorem is one of the earliest probabilistic inference algorithms developed by Reverend Bayes \(which he used to try and infer the existence of God no less\) and still performs extremely well for certain use cases.
>
> It's best to understand this theorem using an example. Let's say you are a member of the Secret Service and you have been deployed to protect the Democratic presidential nominee during one of his/her campaign speeches. Being a public event that is open to all, your job is not easy and you have to be on the constant lookout for threats. So one place to start is to put a certain threat-factor for each person. So based on the features of an individual, like the age, sex, and other smaller factors like is the person carrying a bag?, does the person look nervous? etc. you can make a judgement call as to if that person is viable threat.
>
> If an individual ticks all the boxes up to a level where it crosses a threshold of doubt in your mind, you can take action and remove that person from the vicinity. The Bayes theorem works in the same way as we are computing the probability of an event\(a person being a threat\) based on the probabilities of certain related events\(age, sex, presence of bag or not, nervousness etc. of the person\).
>
> One thing to consider is the independence of these features amongst each other. For example if a child looks nervous at the event then the likelihood of that person being a threat is not as much as say if it was a grown man who was nervous. To break this down a bit further, here there are two features we are considering, age AND nervousness. Say we look at these features individually, we could design a model that flags ALL persons that are nervous as potential threats. However, it is likely that we will have a lot of false positives as there is a strong chance that minors present at the event will be nervous. Hence by considering the age of a person along with the 'nervousness' feature we would definitely get a more accurate result as to who are potential threats and who aren't.
>
> This is the 'Naive' bit of the theorem where it considers each feature to be independant of each other which may not always be the case and hence that can affect the final judgement.
>
> In short, the Bayes theorem calculates the probability of a certain event happening\(in our case, a message being spam\) based on the joint probabilistic distributions of certain other events\(in our case, a message being classified as spam\). We will dive into the workings of the Bayes theorem later in the mission, but first, let us understand the data we are going to work with.

#### a introduction of bag of words

> What we have here in our data set is a large collection of text data \(5,572 rows of data\). Most ML algorithms rely on numerical data to be fed into them as input, and email/sms messages are usually text heavy.
>
> Here we'd like to introduce the Bag of Words\(BoW\) concept which is a term used to specify the problems that have a 'bag of words' or a collection of text data that needs to be worked with. The basic idea of BoW is to take a piece of text and count the frequency of the words in that text. It is important to note that the BoW concept treats each word individually and the order in which the words occur does not matter.
>
> Using a process which we will go through now, we can covert a collection of documents to a matrix, with each document being a row and each word\(token\) being the column, and the corresponding \(row,column\) values being the frequency of occurrance of each word or token in that document.
>
> Lets break this down and see how we can do this conversion using a small set of documents.
>
> To handle this, we will be using sklearns[count vectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer) method which does the following:
>
> * It tokenizes the string\(separates the string into individual words\) and gives an integer ID to each token.
> * It counts the occurrance of each of those tokens.
>
> \*\* Please Note: \*\*
>
> * The CountVectorizer method automatically converts all tokenized words to their lower case form so that it does not treat words like 'He' and 'he' differently. It does this using the `lowercase` parameter which is by default set to `True`.
> * It also ignores all punctuation so that words followed by a punctuation mark \(for example: 'hello!'\) are not treated differently than the same words not prefixed or suffixed by a punctuation mark \(for example: 'hello'\). It does this using the `token_pattern` parameter which has a default regular expression which selects tokens of 2 or more alphanumeric characters.
> * The third parameter to take note of is the `stop_words` parameter. Stop words refer to the most commonly used words in a language. They include words like 'am', 'an', 'and', 'the' etc. By setting this parameter value to `english`, CountVectorizer will automatically ignore all words\(from our input text\) that are found in the built in list of english stop words in scikit-learn. This is extremely helpful as stop words can skew our calculations when we are trying to find certain key words that are indicative of spam.
>
> We will dive into the application of each of these into our model in a later step, but for now it is important to be aware of such preprocessing techniques available to us when dealing with textual data.

Contain the implementation of the Bag of Words process from scratch

Potential issues of Bags of Words and the solutions:

> One potential issue that can arise from using this method out of the box is the fact that if our dataset of text is extremely large\(say if we have a large collection of news articles or email data\), there will be certain values that are more common that others simply due to the structure of the language itself. So for example words like 'is', 'the', 'an', pronouns, grammatical contructs etc could skew our matrix and affect our analyis.
>
> There are a couple of ways to mitigate this. One way is to use the `stop_words` parameter and set its value to `english`. This will automatically ignore all words\(from our input text\) that are found in a built in list of English stop words in scikit-learn.
>
> Another way of mitigating this is by using the [tfidf](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) method. This method is out of scope for the context of this lesson.

What does the term 'Naive' in 'Naive Bayes' mean?

> The term 'Naive' in Naive Bayes comes from the fact that the algorithm considers the features that it is using to make the predictions to be independent of each other, which may not always be the case. So in our Diabetes example, we are considering only one feature, that is the test result. Say we added another feature, 'exercise'. Let's say this feature has a binary value of `0` and `1`, where the former signifies that the individual exercises less than or equal to 2 days a week and the latter signifies that the individual exercises greater than or equal to 3 days a week. If we had to use both of these features, namely the test result and the value of the 'exercise' feature, to compute our final probabilities, Bayes' theorem would fail. Naive Bayes' is an extension of Bayes' theorem that assumes that all the features are independent of each other.

Naive Bayes: multinomial vs Gaussian

> Specifically, we will be using the multinomial Naive Bayes implementation. This particular classifier is suitable for classification with discrete features \(such as in our case, word counts for text classification\). It takes in integer word counts as its input. On the other hand Gaussian Naive Bayes is better suited for continuous data as it assumes that the input data has a Gaussian\(normal\) distribution.

Model evaluation:

> \*\* Accuracy \*\* measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions \(the number of test data points\).
>
> \*\* Precision \*\* tells us what proportion of messages we classified as spam, actually were spam.It is a ratio of true positives\(words classified as spam, and which are actually spam\) to all positives\(all words classified as spam, irrespective of whether that was the correct classification\), in other words it is the ratio of
>
> `[True Positives/(True Positives + False Positives)]`
>
> \*\* Recall\(sensitivity\)\*\* tells us what proportion of messages that actually were spam were classified by us as spam.It is a ratio of true positives\(words classified as spam, and which are actually spam\) to all the words that were actually spam, in other words it is the ratio of
>
> `[True Positives/(True Positives + False Negatives)]`
>
> For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam\(including the 2 that were spam but we classify them as not spam, hence they would be false negatives\) and 10 as spam\(all 10 false positives\) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score.We will be using all 4 metrics to make sure our model does well. For all 4 metrics whose values can range from 0 to 1, having a score as close to 1 as possible is a good indicator of how well our model is doing.
>
> We will be using all 4 metrics to make sure our model does well. For all 4 metrics whose values can range from 0 to 1, having a score as close to 1 as possible is a good indicator of how well our model is doing.
>
> ```python
> from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
> print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
> print('Precision score: ', format(precision_score(y_test, predictions)))
> print('Recall score: ', format(recall_score(y_test, predictions)))
> print('F1 score: ', format(f1_score(y_test, predictions)))
> ```

Advantage of Naive Bayes

> One of the major advantages that Naive Bayes has over other classification algorithms is its ability to handle an extremely large number of features. In our case, each word is treated as a feature and there are thousands of different words. Also, it performs well even with the presence of irrelevant features and is relatively unaffected by them. The other major advantage it has is its relative simplicity. Naive Bayes' works well right out of the box and tuning it's parameters is rarely ever necessary, except usually in cases where the distribution of the data is known. It rarely ever overfits the data. Another important advantage is that its model training and prediction times are very fast for the amount of data it can handle. All in all, Naive Bayes' really is a gem of an algorithm!

prerequisite statistics courses:[https://www.udacity.com/course/intro-to-inferential-statistics--ud201](https://www.udacity.com/course/intro-to-inferential-statistics--ud201)[https://www.udacity.com/course/intro-to-descriptive-statistics--ud827](https://www.udacity.com/course/intro-to-descriptive-statistics--ud827)

median is better than mean as median is not affected severely by outliers.

Bessel's correction is the use of n − 1 instead of n in the formula for the sample variance and sample standard deviation, where n is the number of observations in a sample. This method corrects the bias in the estimation of the population variance \(the sample variance and sample standard deviation tends to be higher if divided by n as the samples will probably nearly the mean\). It also partially corrects the bias in the estimation of the population standard deviation. However, the correction often increases the mean squared error in these estimations.

Label Encoder vs One-hot Encoder:

One thing to keep in mind when encoding data is the fact that you do not want to skew your analysis because of the numbers that are assigned to your categories. For example, in the above example, slim is assigned a value 2 and obese a value 1. This is not to say that the intention here is to have slim be a value that is empirically twice is likely to affect your analysis as compared to obese. In such situations it is better to one-hot encode your data as all categories are assigned a 0 or a 1 value thereby removing any unwanted biases that may creep in if you simply label encode your data.

If we have concerns about class imbalance, then we can use the StratifiedKFold\(\) class instead. Where KFold\(\) assigns points to folds without attention to output class, StratifiedKFold\(\) assigns data points to folds so that each fold has approximately the same number of data points of each output class. This is most useful for when we have imbalanced numbers of data points in your outcome classes \(e.g. one is rare compared to the others\).

K-fold cross-validation training technique is a cross-validation training technique which randomly partition the original sample into k equal sized subsamples and of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times \(the folds\), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged to produce a single estimation.The benefit this technique provides for grid search when optimizing a model is that is that all observations are used for both training and validation, and each observation is used for validation exactly once and thus it matters less how the data gets divided and the variance of the resulting estimate is reduced as k is increased.

GridSearchCV: Each of the combination in the grid is used to train an SVM, and the performance is then assessed using cross-validation.

Evaluation:Accuracy = no. of all data points labeled correctly divided by all data pointsShortcoming:

* not ideal for skewed classes
* not meet your need if you want to err on one label \(err on the side of guilty as all the selected person will be further investigated \(not putting to jail directly\)\)

Precision: True Positive / \(True Positive + False Positive\). Out of all the items labeled as positive, how many truly belong to the positive class.Somehow you take a few shots and if most of them got their target \(relevant documents\) then you have a high precision, regardless of how many shots you fired \(number of documents that got retrieved\).

Recall: True Positive / \(True Positive + False Negative\). Out of all the items that are truly positive, how many were correctly classified as positive. Or simply, how many positive items were 'recalled' from the dataset.Recall is the fraction of the documents that are relevant to the query that are successfully retrieved, hence its name \(in English recall = the action of remembering something\).

Confusion Matrix

The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0:F1 = 2 \* \(precision \* recall\) / \(precision + recall\)

Popular regression metrics are:

Mean Absolute Error

One way to measure error is by using absolute error to find the predicted distance from the true value. The mean absolute error takes the total absolute error of each example and averages the error based on the number of data points. By adding up all the absolute values of errors of a model we can avoid canceling out errors from being too high or below the true values and get an overall error metric to evaluate the model on.

Mean Squared Error

Mean squared is the most common metric to measure model performance. In contrast with absolute error, the residual error \(the difference between predicted and the true value\) is squared.

Some benefits of squaring the residual error is that error terms are positive, it emphasizes larger errors over smaller errors, and is differentiable. Being differentiable allows us to use calculus to find minimum or maximum values, often resulting in being more computationally efficient.

In addition to error metrics, scikit-learn contains two scoring metrics which scale continuously from 0 to 1, with values of 0 being bad and 1 being perfect performance.

These are the metrics that you'll use in the project at the end of the course. They have the advantage of looking similar to classification metrics, with numbers closer to 1.0 being good scores and bad scores tending to be near 0.

One of these is the R2 score, which computes the coefficient of determination of predictions for true values. This is the default scoring method for regression learners in scikit-learn.

The other is the explained variance score.

To learn more about bias and variance, we recommend this [essay](http://scott.fortmann-roe.com/docs/BiasVariance.html) by Scott Fortmann-Roe.

Curse of Dimensionality: As the number of features or dimensions grows, the amount of data we need to generalizes accurately grows exponentially.

Regression:

[https://www.mathsisfun.com/algebra/polynomials.html](https://www.mathsisfun.com/algebra/polynomials.html)

Besides parametric \(using a polynomial\) can also be non-parametric \(data-centric approach / instance-based approach, e.g. kNN\)

Why is Deep Learning taking off?

The loss function computes the error for a single training example, the cost function is the average of the loss function of the entire training set.

Derivative is just slope.

Vectorization is key in the deep learning era.

Neural network programming guideline

* whenever possible, avoid explicit for-loops-

Use

```text
 a = np.random.randn(5,1)
```

instead of

```text
 a = np.random.randn(5)
```

The 2nd one creates a "rank 1 array".

"gradient \(also called the slope or derivative\) of the sigmoid function with respect to its input x"So gradient = slope = derivative of a function

What you need to remember:

* np.exp\(x\) works for any np.array x and applies the exponential function to every coordinate
* the sigmoid function and its gradient
* image2vector is commonly used in deep learning
* np.reshape is widely used. In the future, you'll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs.
* numpy has efficient built-in functions
* broadcasting is extremely useful

What to remember:

* Vectorization is very important in deep learning. It provides computational efficiency and clarity.
* You have reviewed the L1 and L2 loss.
* You are familiar with many numpy functions such as np.sum, np.dot, np.multiply, np.maximum, etc...

What you need to remember:Common steps for pre-processing a new dataset are:

* Figure out the dimensions and shapes of the problem \(m\_train, m\_test, num\_px, ...\)
* Reshape the datasets such that each example is now a vector of size \(num\_px \* num\_px \* 3, 1\)
* "Standardize" the data

> ## FORWARD PROPAGATION \(FROM X TO COST\)
>
> ## BACKWARD PROPAGATION \(TO FIND GRAD\)

> A = None \# compute activationSo A is the activation?

## 

## Log Loss

### From [Exegetic Analytics](http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/):

Logarithmic Loss, or simply Log Loss, is a [classification loss function](https://en.wikipedia.org/wiki/Loss_functions_for_classification) often used as an evaluation metric in [kaggle](https://www.kaggle.com/wiki/MultiClassLogLoss) competitions. Since success in these competitions hinges on effectively minimising the Log Loss, it makes sense to have some understanding of how this metric is calculated and how it should be interpreted.

Log Loss quantifies the accuracy of a classifier by penalising false classifications. Minimising the Log Loss is basically equivalent to maximising the accuracy of the classifier, but there is a subtle twist which we’ll get to in a moment.

In order to calculate Log Loss the classifier must assign a probability to each class rather than simply yielding the most likely class. Mathematically Log Loss is defined as

$$\frac{1}{N} \sum\\_{i=1}^N \sum\\_{j=1}^M y\\_{ij} \log \, p\\_{ij}​$$​

where N is the number of samples or instances, M is the number of possible labels, ​ is a binary indicator of whether or not label ​ is the correct classification for instance ​, and ​ is the model probability of assigning label ​ to instance ​. A perfect classifier would have a Log Loss of precisely zero. Less ideal classifiers have progressively larger values of Log Loss. If there are only two classes then the expression above simplifies to

​$$\frac{1}{N} \sum\\_{i=1}^N \[y\\_{i} \log \, p\\_{i} + \(1 - y\\_{i}\) \log \, \(1 - p\\_{i}\)\]​$$

Note that for each instance only the term for the correct class actually contributes to the sum.

#### Log Loss Function

Let’s consider a simple implementation of a Log Loss function:

```r
> LogLossBinary = function(actual, predicted, eps = 1e-15) {
+ predicted = pmin(pmax(predicted, eps), 1-eps)
+ - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
+ }
```

Suppose that we are training a binary classifier and consider an instance which is known to belong to the target class. We’ll have a look at the effect of various predictions for class membership probability.

```r
> LogLossBinary(1, c(0.5))  
[1] 0.69315  
> LogLossBinary(1, c(0.9))  
[1] 0.10536  
> LogLossBinary(1, c(0.1))  
[1] 2.3026
```

In the first case the classification is neutral: it assigns equal probability to both classes, resulting in a Log Loss of 0.69315. In the second case the classifier is relatively confident in the first class. Since this is the correct classification the Log Loss is reduced to 0.10536. The third case is an equally confident classification, but this time for the wrong class. The resulting Log Loss escalates to 2.3026. Relative to the neutral classification, being confident in the wrong class resulted in a far greater change in Log Loss. Obviously the amount by which Log Loss can decrease is constrained, while increases are unbounded.

#### Looking Closer

Let’s take a closer look at this relationship. The plot below shows the Log Loss contribution from a single positive instance where the predicted probability ranges from 0 \(the completely wrong prediction\) to 1 \(the correct prediction\). It’s apparent from the gentle downward slope towards the right that the Log Loss gradually declines as the predicted probability improves. Moving in the opposite direction though, the Log Loss ramps up very rapidly as the predicted probability approaches 0. That’s the twist I mentioned earlier.

![](.gitbook/assets/image%20%281%29.png)

&lt;br&gt; Log Loss heavily penalises classifiers that are confident about an incorrect classification. For example, if for a particular observation, the classifier assigns a very small probability to the correct class then the corresponding contribution to the Log Loss will be very large indeed. Naturally this is going to have a significant impact on the overall Log Loss for the classifier. The bottom line is that it’s better to be somewhat wrong than emphatically wrong. Of course it’s always better to be completely right, but that is seldom achievable in practice! There are at least two approaches to dealing with poor classifications:

1. Examine the problematic observations relative to the full data set. Are they simply outliers? In this case, remove them from the data and re-train the classifier.
2. Consider smoothing the predicted probabilities using, for example, [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing). This will result in a less “certain” classifier and might improve the overall Log Loss.

#### Code Support for Log Loss

Using Log Loss in your models is relatively simple. [XGBoost](https://github.com/dmlc/xgboost) has `logloss` and `mlogloss` options for the `eval_metric` parameter, which allow you to optimise your model with respect to binary and multiclass Log Loss respectively. Both metrics are available in [caret](http://topepo.github.io/caret/index.html)’s `train()` function as well. The [Metrics](https://cran.r-project.org/web/packages/Metrics/index.html) package also implements a number of Machine Learning metrics including Log Loss.

### From [scikit-learn](http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss):

> &gt; Log loss, also called logistic regression loss or cross-entropy loss, is defined on probability estimates. It is commonly used in \(multinomial\) logistic regression and neural networks, as well as in some variants of expectation-maximization, and can be used to evaluate the probability outputs \(\`predict\_proba\`\) of a classifier instead of its discrete predictions.
>
> &gt;
>
> &gt; For binary classification with a true label $y \in \{0,1\}$ and a probability estimate $p = \operatorname{Pr}\(y = 1\)$, the log loss per sample is the negative log-likelihood of the classifier given the true label:
>
> &gt;
>
> &gt; $$L\_{\log}\(y, p\) = -\log \operatorname{Pr}\(y\|p\) = -\(y \log \(p\) + \(1 - y\) \log \(1 - p\)\)$$
>
> &gt;
>
> &gt; This extends to the multiclass case as follows. Let the true labels for a set of samples be encoded as a 1-of-K binary indicator matrix $Y$, i.e., $y\\_{i,k} = 1$ if sample $i$ has label $k$ taken from a set of $K$ labels. Let $P$ be a matrix of probability estimates, with $p\\_{i,k} = \operatorname{Pr}\(t\_{i,k} = 1\)$. Then the log loss of the whole set is
>
> &gt;
>
> &gt; $$L\\_{\log}\(Y, P\) = -\log \operatorname{Pr}\(Y\|P\) = - \frac{1}{N} \sum\\_{i=0}^{N-1} \sum\\_{k=0}^{K-1} y\\_{i,k} \log p\\_{i,k}$$
>
> &gt;
>
> &gt; To see how this generalizes the binary log loss given above, note that in the binary case, $p\\_{i,0} = 1 - p\\_{i,1}$ and $y\\_{i,0} = 1 - y\\_{i,1}$, so expanding the inner sum over $y\\_{i,k} \in \{0,1\}$ gives the binary log loss.
>
> &gt;
>
> &gt; The \`log\_loss\` function computes log loss given a list of ground-truth labels and a probability matrix, as returned by an estimator's \`predict\_proba\` method.

```text
 >>> from sklearn.metrics import log_loss>>> y_true = [0, 0, 1, 1]>>> y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]>>> log_loss(y_true, y_pred)    0.1738...
```

> The first `[.9, .1]` in `y_pred` denotes 90% probability that the first sample has label 0. The log loss is non-negative.

## Idea

In the [TED talk](https://www.ted.com/talks/mallory_soldner_your_company_s_data_could_end_world_hunger/) filmed September 2016, Mallory Soldner, UPS's advanced analytics manager, talked about how can companies do [data philanthropy](https://www.wikiwand.com/en/Data_philanthropy) and why.

### Donating Data

As Mallory said:

> Companies today, they collect mountains of data, so the first thing they can do is start donating that data. Some companies are already doing it. Take, for example, a major telecom company. They opened up their data in Senegal and the Ivory Coast and researchers discovered that if you look at the patterns in the pings to the cell phone towers, you can see where people are traveling. And that can tell you things like where malaria might spread, and you can make predictions with it. Or take for example an innovative satellite company. They opened up their data and donated it, and with that data you could track how droughts are impacting food production. With that you can actually trigger aid funding before a crisis can happen.

### Donating Decision Scientists

As Mallory said:

> But even if the floodgates opened up, and even if all companies donated their data to academics, to NGOs, to humanitarian organizations, it wouldn't be enough to harness that full impact of data for humanitarian goals. Why? To unlock insights in data, you need decision scientists. Decision scientists are people like me. They take the data, they clean it up, transform it and put it into a useful algorithm that's the best choice to address the business need at hand. In the world of humanitarian aid, there are very few decision scientists. Most of them work for companies. So that's the second thing that companies need to do.

### Donating Technology to Gather New Source of Data

As Mallory said:

> Right now, Syrian refugees are flooding into Greece, and the UN refugee agency, they have their hands full. The current system for tracking people is paper and pencil, and what that means is that when a mother and her five children walk into the camp, headquarters is essentially blind to this moment. That's all going to change in the next few weeks, thanks to private sector collaboration. There's going to be a new system based on donated package tracking technology from the logistics company that I work for. With this new system, there will be a data trail, so you know exactly the moment when that mother and her children walk into the camp. And even more, you know if she's going to have supplies this month and the next. Information visibility drives efficiency. For companies, using technology to gather important data, it's like bread and butter. They've been doing it for years, and it's led to major operational efficiency improvements. Just try to imagine your favorite beverage company trying to plan their inventory and not knowing how many bottles were on the shelves. It's absurd. Data drives better decisions.

### Why Companies Should Do These

As Mallory said:

> Well for one thing, beyond the good PR, humanitarian aid is a 24-billion-dollar sector, and there's over five billion people, maybe your next customers, that live in the developing world.

> Further, companies that are engaging in data philanthropy, they're finding new insights locked away in their data. Take, for example, a credit card company that's opened up a center that functions as a hub for academics, for NGOs and governments, all working together. They're looking at information in credit card swipes and using that to find insights about how households in India live, work, earn and spend. For the humanitarian world, this provides information about how you might bring people out of poverty. But for companies, it's providing insights about your customers and potential customers in India. It's a win all around.

> Now, for me, what I find exciting about data philanthropy — donating data, donating decision scientists and donating technology — it's what it means for young professionals like me who are choosing to work at companies. Studies show that the next generation of the workforce care about having their work make a bigger impact. We want to make a difference, and so through data philanthropy, companies can actually help engage and retain their decision scientists. And that's a big deal for a profession that's in high demand.

## Python

### Key notes for Pandas

Why Pandas DataFrame but not 2D Numpy array? Because Numpy array can only contain one data type.

Pandas is built on Numpy.

In a simplified sense, you can think of series as a 1D labelled array.

### pd.read\_csv

From [https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words):

```text
 # Import the pandas package, then use the "read_csv" function to read# the labeled training dataimport pandas as pd       train = pd.read_csv("labeledTrainData.tsv", header=0, \                    delimiter="\t", quoting=3)
```

> Here, "header=0" indicates that the first line of the file contains column names, "delimiter=\t" indicates that the fields are separated by tabs, and quoting=3 tells Python to ignore doubled quotes, otherwise you may encounter errors trying to read the file.

### Context manager

From \[D\]ataCamp\]\([https://campus.datacamp.com/courses/python-data-science-toolbox-part-2/bringing-it-all-together-3?ex=7](https://campus.datacamp.com/courses/python-data-science-toolbox-part-2/bringing-it-all-together-3?ex=7)\):

> The csv file 'world\_dev\_ind.csv' is in your current directory for your use. To begin, you need to open a connection to this file using what is known as a context manager. For example, the command with open\('datacamp.csv'\) as datacamp binds the csv file 'datacamp.csv' as datacamp in the context manager. Here, the with statement is the context manager, and its purpose is to ensure that resources are efficiently allocated when opening a connection to a file.
>
> If you'd like to learn more about context managers, refer to the [DataCamp course on Importing Data in Python](https://campus.datacamp.com/courses/python-data-science-toolbox-part-2/bringing-it-all-together-3?ex=7).

```text
 # Open a connection to the filewith open('world_dev_ind.csv') as file:​    # Skip the column names    file.readline()​    # Initialize an empty dictionary: counts_dict    counts_dict = {}​    # Process only the first 1000 rows    for j in range(0,1000):​        # Split the current line into a list: line        line = file.readline().split(',')​        # Get the value for the first column: first_col        first_col = line[0]​        # If the column value is in the dict, increment its value        if first_col in counts_dict.keys():            counts_dict[first_col] += 1​        # Else, add to the dict and set value to 1        else:            counts_dict[first_col] = 1<script.py> output:    {'Euro area': 119, 'Arab World': 80, 'Caribbean small states': 77, 'East Asia & Pacific (all income levels)': 122, 'Fragile and conflict affected situations': 76, 'Europe & Central Asia (developing only)': 89, 'Central Europe and the Baltics': 71, 'Heavily indebted poor countries (HIPC)': 18, 'Europe & Central Asia (all income levels)': 109, 'East Asia & Pacific (developing only)': 123, 'European Union': 116}
```

### Select Data

Data can be selected from Pandas DataFrame using:

1. Square brackets
2. the `loc` method, which is label-based \(i.e. using the labels of columns and observations\)\(inclusive\)
3. the `iloc` method, which is position-based \(i.e. using the index of columns and observations\)\(exlcusive\)

Below are some examples

```text
 # Print out country column as Pandas Seriesprint(cars['country'])​# Print out country column as Pandas DataFrameprint(cars[['country']])​# Print out DataFrame with country and drives_right columnsprint(cars[['country','drives_right']])​# Print out first 3 observationsprint(cars[:3])​# Print out fourth, fifth and sixth observationprint(cars[3:6])​# Print out drives_right column as Seriesprint(cars.loc[:, 'drives_right'])​# Print out drives_right column as DataFrameprint(cars.loc[:,['drives_right']])​# Print out cars_per_cap and drives_right as DataFrameprint(cars.loc[:,['cars_per_cap','drives_right']])
```

The `loc` and `iloc` method are more powerful but square brackets are good enough for selecting all observations of some columns.

### 

