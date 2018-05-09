# Inbox

{% embed data="{\"url\":\"https://www.leiphone.com/news/201711/O7rueJQvK21TwHpA.html\",\"type\":\"link\",\"title\":\"北大数字金融研究中心：2017美国Fintech产业六大特征，八大启示（附原文下载）  \| 雷锋网\",\"description\":\"考察报告主要分为两个部分，第一部分是美国金融科技发展现状，第二部分是对于中国的借鉴意义。\",\"icon\":{\"type\":\"icon\",\"url\":\"https://www.leiphone.com/favicon.ico\",\"aspectRatio\":0}}" %}



[5 Reasons “Logistic Regression” should be the first thing you learn when becoming a Data Scientist](https://towardsdatascience.com/5-reasons-logistic-regression-should-be-the-first-thing-you-learn-when-become-a-data-scientist-fcaae46605c4)

## To be categorized

[Udacity Career Development resources](https://docs.google.com/document/d/1Bf1jEkKlCYJJdUoyBSROxPS5FsZlAaZ8cZ35GsHhpYU/pub?embedded=true)

## fast.ai

Using `shift-tab` to check the details of a method.Click `H` to get shortcutEmbedding is a shortcut to a one-hot encoding follows by a matrix product.

## JP Morgan

### Deep Learning architectures

* Multilayer Perceptron \(MLP\) is one of the first designs of multi-layer neural networks, designed in such a way that the input signal passes through each node of the network only once \(also known as a ‘feed-forward’ network\).
* Long Short-term memory \(LSTM\) is a neural network architecture that includes feedback loops between elements. This can also simulate memory, by passing the previous signals through the same nodes. LSTM neural networks are suitable for time series analysis, because they can more effectively recognize patterns and regimes across different time scales.
* Convolutional Neural Networks \(CNN\) are often used for classifying images. They extract data features by passing multiple filters over overlapping segments of the data \(this is related to the mathematical operation of convolution\).
* Restricted Boltzmann Machine \(RBM\) is a neural-network based dimensionality reduction \(unsupervised learning\) technique. The neurons in RBM form two layers called the visible units \(reflecting returns of assets\) and the hidden units \(reflecting latent factors\). Neurons within a layer – hidden or visible – are not connected; this is the ‘restriction’ in the restricted Boltzman machine.

## Deep Learning vs. Classical Machine Learning

* Despite such successes, Deep Learning tools are rarely used in time series analysis, where tools from ‘classical’ Machine Learning dominate.
* Anecdotal evidence from observing winning entries at data science competitions \(like Kaggle\) suggests that structured data is best analyzed by tools like XGBoost and Random Forests. Use of Deep Learning in winning entries is limited to analysis of images or text. Deep Learning tools still require a substantial amount of data to train. Training on small sample sizes \(through so-called generative-adversarial models\) is still at an incipient research stage. The necessity of having large sample data implies that one may see application of Deep Learning to intraday or high-frequency trading before we see its application in lower frequencies.
* Deep Learning finds immediate use for portfolio managers in an indirect manner. Parking lot images are analyzed using Deep Learning architectures \(like convolutional neural nets\) to count cars. Text in social media is analyzed using Deep Learning architectures \(like long short-term memory\) to detect sentiment. Such traffic and sentiment signals can be integrated directly into quantitative strategies, as shown in earlier sections of this report. Calculation of such signals themselves will be outsourced to specialized firms that will design bespoke neural network architecture for the task.

## Time-Series Analysis: Long Short-Term Memory

* While LSTM is designed keeping in mind such time-series, there is little research available on its application to econometrics or financial time-series. After describing the basic architecture of an LSTM network below, we also provide a preliminary example of a potential use of LSTM in forecasting asset prices.
* We attempted to use an LSTM neural network to design an S&P 500 trading strategy. The dataset included S&P 500 returns since 2000, and the first 14 years were used for training the LSTM. The last 3 years worth of data were used to predict monthly returns.

## Cheatsheet

[Essential Cheat Sheets for Machine Learning and Deep Learning Engineers](https://startupsventurecapital.com/essential-cheat-sheets-for-machine-learning-and-deep-learning-researchers-efb6a8ebd2e5)

## Notes from ND009

### algorithm mentioned

#### supervised learning

* decision trees
* naive Bayes
* Gradient descent \("will be used extensively"\)
* linear regression
  * \(the procedure of finding the best line is actually a gradient descent by moving the direction where the error is reduced\)
  * the procedure is called "least square" \(minimizing the error, in square to handle the negative\)
* logistic regression \(gradient descent with log loss as error function\)
* support vector machine \(_what is the difference compared to logistic regression?_\)
* neural network
* kernel trick - well used in support vector machine \(so not an algorithms?\) \(find a function that separate the 2 target variables and define it as an additional feature\)

I should be able to explain the algorithms and know the pros and cons for each, and what are their parameters

algorithms mentioned by [Accenture Finance & Risk Data Scientist Consultant](https://www.linkedin.com/jobs/view/356253635/):k-NN, Naive Bayes, SVM, Random Forestsso at least need to understand these

logistic regression vs SVM

* logistic regression considers all the data points
* SVM just consider the one closest to the boundary

#### unsupervised learning

* k-mean clustering \(limitation, need to know the number of cluster in advance\)
* hierarchical clustering \(limitation, need to predefine the max distance which we stop grouping\)

#### other

one-line definition of supervised learning:feeding a labelled dataset into the model, that it can learn from, to make future predictions

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

## Get a job

### Concept

[The 3 Stages of Data Science: Dashboards, Machine Learning, Actions](http://101.datascience.community/2017/08/09/the-3-stages-of-data-science/)

AutoML:[When asked if they are on track to put themselves out of a job, Le and Zoph laugh, though. Right now the technique is too expensive to be widely used. The pair's experiments tied up 800 powerful graphics processors for multiple weeks—racking up the kind of power bill few companies could afford for speculative research.](https://www.technologyreview.com/s/607894/why-googles-ceo-is-excited-about-automating-artificial-intelligence/)

[Why China may lead AI](https://www.jiqizhixin.com/articles/2017-07-25-3):

* 計算能力以及資本：從阿里巴巴與騰訊這樣的巨頭，到 CIB FinTech 與 UCloud 這樣的初創公司，這些中國企業都在加快速度建立數據中心。據諮詢公司 Gartner 報導，雲計算市場近年來已增長了 30% 之多，且將持續增長下去。據智囊團烏鎮智庫統計，2012 - 2016 年，中國的人工智能企業獲得了 26 億美元資金，雖然低於美國同行的 179 億美元，但總體而言仍在飛速增長。
* 研究型人才：在阿里巴巴負責管理 150 位數據科學家的閔萬里說，在中國發現頂尖的人工智能專家要難於美國。但他預測，由於許多大學都推出了人工智能計畫，所以未來幾年將有所改變。據估計，中國擁有超過世界五分之二的高素質人工智能科學家。
* 數據：中國的人口等規模及數據多樣性為這一循環提供了有力的燃料。僅靠日常生活，全國近 1.4 億人口所產生的數據便多於其他幾個國家之和。即便是一些罕見病，也不曾缺乏用來教算法識別這種病的病例。由於打漢字比西方國家的文字更為費力，因此中國人往往比西方人更傾向於使用語音識別服務，於是公司便擁有更多語音片段來改進語音產品。
  * 7.3 億的互聯網用戶
  * 中國人似乎並不十分重視隱私
  * 中國的年輕人似乎十分熱衷於以人工智能驅動的服務，對個人數據的使用也很放鬆。
* 政府支持：在中國，人工智能的另一個重要支持者便是政府，這項技術在其目前的五年計畫中地位顯赫。技術公司正與政府機構進行密切合作，如百度已響應號召，領導國家深度學習實驗室。但政府用清規戒律對人工智能公司加壓的可能性微乎其微，中國包含個人資料保護相關規定的法律有 40 多條，但它們鮮少得以執行。

### interview Questions

[21 Must-Know Machine Learning Interview Questions and Answers](https://elitedatascience.com/machine-learning-interview-questions-answers)

作者：晓宇我喜欢你链接：[https://www.zhihu.com/question/23259302/answer/24300412](https://www.zhihu.com/question/23259302/answer/24300412)来源：知乎著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

我面試過5-6家互聯網公司的數據挖掘和分析、機器學習相關職位的工程師。被問到下面一些問題。SVM的原理，SVM裡面的核K-means，如何用hadoop實現k-meansnaive bayes和logistic regression的區別LDA的原理和推導做廣告點擊率預測，用哪些數據什麼算法推薦系統的算法中最近鄰和矩陣分解各自適用場景用戶流失率預測怎麼做（遊戲公司的數據挖掘都喜歡問這個）一個遊戲的設計過程中該收集什麼數據如何從登陸日誌中挖掘儘可能多的信息這些問題我回答的情況，分幾種。一種是在面試官的提示下，算是勉強完成了答案。一種是在面試官的提示下，答了一點但是答得不夠好。一種是面試官不提示也沒有反饋，我回答了但是我不知道回答得怎樣。我非常後悔的一點是我現在才想起來總結。有一個題是遊戲玩家流失率預測，我被問過兩次。但是每次我都說是個分類問題。最近我突然想起來去網上查了下，有兩個點，數據不平衡問題和時間序列分析。我網上查到是一個大學教授和人人遊戲合作的課題。我然後查了下這個老師的publication。沒發現相關的論文。可能公司不讓發表吧。這些問題的特點是很基礎很簡單，因為實際中很少用複雜的算法，複雜的算法不好控制，而且理論要求高。另一個特點是注重考查實際工程能力，我經常被問到自己實現了哪些算法。還有的問題很契合實際。我覺得如果現在再給我準備的機會。我會準備下面幾點。首先是計算機基礎知識和算法，這些都是會正常考察的。有些公司考的少，有些公司正常考察。針對機器學習這部分，需要理論紮實，還需要自己動手實現代碼。另外hadoop，mpi，最近比較火的spark，應該都是加分項。另一個是接觸下實際的數據分析系統。我在學校裡面看的論文，都是講算法的多，講應用系統的少。這個可以靠之前的實習，也可以看些比較實用的論文。

### Trend

一个完美的物联网平台需要考虑 5 个要素

* 向混合应用环境转变。既要给出简单的、拿来就能用的应用，也要提供定制服务。首先，越简单的应用越受欢迎。同时，平台不一定能向商家一样了解他们的业务，因此确保应用的开发友好程度也很重要。
* 注意提取和整合数据的能力。数据是驱动物联网产业最重要的燃料，也是物联网平台的意义，因此，请保证你的平台具有管理大量的、高速的、多来源数据的能力。
* 注意云端基础设施的兼容性。选择与一家大型基础设施提供商（IaaS）合作，也意味着选择了他们的其他配套服务。有一些小型的 IoT 玩家可能只选择一个或有限的几个云提供商，但是请确保你的 IoT 平台和你的企业云平台是兼容的。
* 数据的所有权和安全性。数据的存储位置和处理位置对于 IoT 平台来说是十分重要的。
* 边缘处理和控制。IoT 平台可以是集中式的，也可以支持边缘计算以减少延迟。有时候，将数据在云端和本地移动的通信成本很高，例如从偏远地区的矿井或航船上，几乎没办法传递 TB 级别的数据，这时候就要考虑边缘计算了。
* 近年來，國外智能音箱市場的競爭已經逐漸趨於白熱化，國內前沿玩家也紛紛入局。智能音箱作為下一代交互的重要載體，背後離不開自然語言理解與語音識別技術的扶持。雖然眼下這兩項技術還處於初期發展階段，不過鑑於可預見的商業價值，各家公司會在該領域進行持續投入並加大扶持力度，預計相關技術將在 3 到 5 年內邁向下一階段，得到實質性進展。
* 目前，虛擬助理、機器學習平台以及優化硬體處在爬升階段，決策管理則處於平穩發展階段，預計 5 到 10 年內仍將是人工智慧領域的熱門方向，具有巨大的研究空間。
* 深度學習平台、語音分析技術、生物識別技術、圖像與視頻分析技術、機器人自動處理系統與文本分析和 NLP 領域在接下來仍將穩步發展，但可能不會有較大突破。其中語義分析技術發展可能較為緩慢，在 5 至 10 年內才能達到上升階段，機器人技術可能在短期內也不會有較大突破。相比之下，圖像及視頻識別和文本分析及 NLP 技術發展較為迅速，預計在 1 至 3 年內就能分別階躍至成長期及平穩期。
* 集群智能技術仍在萌芽期，值得長期研究，具有巨大的發展潛能。

目前的机器学习，还存在不少技术瓶颈：\(1\)尽管在某些点有重大突破，但在更多的领域只能处于实验室研究阶段。比如，面部识别在实验室测试时，识别率可以达到98%以上。然而如果你把一台摄像机放到大街上，识别率能达到40%已经是非常不错。\(2\) 需要海量的学习数据。两三岁的小孩，只要大人给他指过几次小狗，下次十有八九他都能认出来。然而不论是多强的机器学习模型，也不可能只看几张小狗的图片，就能准确地认识小狗。\(3\) 想学习机器学习，需要技术人员学习大量的基础知识和算法。

在金融 AI 需求旺盛的大场景下，传统银行以及金融机构对于人工智能技术的布局却普遍稍晚。因此，他们倾向于向互联网公司寻求合作。今年 6 月，国有四大行中的三家——中国工商银行、中国农业银行、中国银行分别与京东金融、百度、腾讯签署合作协议，侧重在新兴金融技术领域的优势互补，如人工智能、区块链、虚拟货币等方面。中国建设银行相比于其他三家行动更早一些，今年 3 月，建行便与阿里巴巴、蚂蚁金服宣布战略合作。按照协议和业务合作备忘录，双方将共同推进建行信用卡线上开卡业务、整合线下线上渠道业务合作、加强电子支付业务合作、打通信用体系。未来，双方还将实现二维码支付互认互扫，支付宝将支持建行手机银行 APP 支付。

[胡一天專欄：金融科技投資漫談](http://www.storm.mg/article/57443)

#### Current issues of FS industry

* 传统系统的遗留问题
* 机构间缺乏统一的愿景
* 没有技术和专家的支持
* 缺乏预算、没有明确的数字化方向以及找不到合适的独立的技术合作伙伴

#### Trend in FS

* 数据分析被认为是银行转型过程中不可或缺的颠覆性技术，79% 的银行表示，这项技术正在或将在未来两年内产生重大影响
* 移动及可穿戴技术和开放式应用程序接口技术也在发挥着关键作用。
* 目前银行业正在使用的 AI 技术仅限于欺诈检测和风险识别、自然语言处理两项技术，分别有 14% 和 5% 的银行使用，50% 和 59% 的银行正在考虑之中。对于对话机器人、语音助手相关应用，当下还没有银行部署完毕，但约有 9% 的银行正在进行中。Celent 没有透露是哪家银行正在部署，但据悉，美国银行和美国第一资本投资国际集团位列其中。另外，分别有 10% 的银行正在布局 RPA（机器人流程自动化）以及自然语言生成技术。除此之外，对于这些新兴 AI 技术，大部分银行尚处无计划状态。

### Kaggle

[The Beginner’s Guide to Kaggle](https://elitedatascience.com/beginner-kaggle)

### Coding

[LeetCode](https://leetcode.com/)

## Tool

### Jupyter Notebook

[Useful extension](https://www.douban.com/review/7890354/)

## Questions to explore

By [anokas](https://www.kaggle.com/anokas/quora-question-pairs/data-analysis-xgboost-starter-0-35460-lb):

> I'll be using a naive method for splitting words \(splitting on spaces instead of using a serious tokenizer\)...

What is a tokenizer and how it works?

> I am using the AUC metric since it is unaffected by scaling and similar, so it is a good metric for testing the predictive power of individual features.

Why AUC metric is unaffected by scaling?

> \[Log Loss\] looks at the probabilities themselves and not just the order of the predictions like AUC

Revisit why AUC just looks at the order of the predictions

## Data Visualizaton

[Visual Vocabulary](http://ft-interactive.github.io/visual-vocabulary/)

[Histograms is very sensitive to parameter choices](https://tinlizzie.org/histograms/) - key takeaways:

* it can be used to mislead people with carefully chosen parameters, including bin offset, bin width

## Machine learning vs. statistical modeling

> Machine learning is a subfield of computer science and is closely related to statistics. Both statistics and machine learning have the aim of learning from data and they share many concepts and mathematical tools.
>
> But, unlike statistics, machine learning tends to emphasise building software to make predictions, is often applied to larger datasets, and the techniques used require fewer assumptions about the data or how it was collected. There’s more detail on the differences [here](https://www.analyticsvidhya.com/blog/2015/07/difference-machine-learning-statistical-modeling/).

## Reinforement learning

[80000 Hours](https://80000hours.org/career-reviews/machine-learning-phd):

> Reinforcement learning is important because it’s a promising approach to creating artificial intelligence that could perform well at multiple different tasks rather than the very narrow applicability that most machine learning systems currently have.

## End-to-end Process for Classification Problem Using Text

With refernce to the [kernel by anokas for Quora Question Pair competitions](https://www.kaggle.com/anokas/quora-question-pairs/data-analysis-xgboost-starter-0-35460-lb)

### Exploratory Data Analysis \(EDA\)

* File sizes: compare the file sizes of training set and test test.
* Print the head of both training and test set to observe the sample values of each field.
* Plot some graphs to explore, especially histogram of character count and word count.
* Plot a word cloud to find out what are the most common words.
* Semantic Analysis: Check the usage of different punctuations in dataset.

#### Image Classification Problem

When handling image classification problems, try to answer the following questions:

* What are the distributions of image types?
* Are the images in the same dimension?

### Feature Engineering

* used TF-IDF \(term-frequency-inverse-document-frequency\), i.e. weigh the terms by how uncommon they are, meaning that we care more about rare words existing in both questions than common one. \([TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)\)

### Model induction

* Data rebalancing: "...since we have 37% positive class in our training data, and only 17% in the test data. By re-balancing the data so our training set has 17% positives, we can ensure that XGBoost outputs probabilities that will better match the data on the leaderboard..."
* run XGBoost

### Model evaluation

#### Overfitting

[The Hazards of Predicting Divorce Without Crossvalidation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1622921/#S1title):

> Overfitting can cause extreme overinflation of predictive powers, especially when oversampled extreme groups and small samples are used, as was the case with [Gottman et al. \(1998; n = 60 couples for the prediction analyses\)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1622921/#R12)...

#### What to do with overfitting per fast.ai

1. Add more data
2. Use data augmentation
3. Use architectures that generalize well
4. Add regularization
5. Reduce architecture complexity

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

### Loop over

#### Dictionaries

```text
 pythonistas = {'hugo':'bowne-anderson', 'francis':'castro'}​for key, value in pythonistas.items():    print(key, value)<script.py> output:    francis castro    hugo bowne-anderson
```

#### List

```text
 # areas listareas = [11.25, 18.0, 20.0, 10.75, 9.50]​for area in areas:    print(area)​# to access the index informationfor index, a in enumerate(areas) :    print("room " + str(index) + ": " + str(a))​<script.py> output:    room 0: 11.25    room 1: 18.0    room 2: 20.0    room 3: 10.75    room 4: 9.5
```

#### Numpy array

```text
 # For loop over np_baseballfor x in np.nditer(np_baseball) :    print(x)
```

#### Series

Simply:

```text
 for entry in col:
```

#### DataFrame

```text
 # Iterate over rows of carsfor lab, row in cars.iterrows() :    print(lab)    print(row)​for lab, row in cars.iterrows() :    print(lab + ": " + str(row["cars_per_cap"]))​# Code for loop that adds COUNTRY columnfor lab, row in cars.iterrows() :    cars.loc[lab, "COUNTRY"] = str.upper(row["country"])
```

### Hashing

I encountered the following error message when doing an exercise in DataCamp:

> 'Series' objects are mutable, thus they cannot be hashed

As I am not sure about what does _hash_ mean, I googled the error message about and found a very good explanation from Stack Overflow as follows:

> Hashing is a concept is computer science which is used to create high performance, pseudo random access data structures where large amount of data is to be stored and accessed quickly.
>
> For example, if you have 10,000 phone numbers, and you want store them in an array \(which is a sequential data structure that stores data in contiguous memory locations, and provides random access\), but you might not have the required amount of contiguous memory locations.
>
> So, you can instead use an array of size 100, and use a hash function to map a set of values to same indices, and these values can be stored in a linked list. This provides a performance similar to an array.
>
> Now, a hash function can be as simple as dividing the number with the size of the array and taking the remainder as the index.
>
> For more detail refer to [https://en.wikipedia.org/wiki/Hash\_function](https://en.wikipedia.org/wiki/Hash_function)

### Function

#### Nested function

From [DataCamp](https://campus.datacamp.com/courses/python-data-science-toolbox-part-1/default-arguments-variable-length-arguments-and-scope?ex=7):

> One other pretty cool reason for nesting functions is the idea of a **closure**. This means that the nested or inner function remembers the state of its enclosing scope when called. Thus, anything defined locally in the enclosing scope is available to the inner function even when the outer function has finished execution.

#### Lambda functions

```text
 # Define echo_word as a lambda function: echo_wordecho_word = (lambda word1, echo: word1 * echo)​# Call echo_word: resultresult = echo_word('hey', 5)​# Print resultprint(result)​<script.py> output:    heyheyheyheyhey
```

**Map\(\) and lambda functions**

```text
 # Create a list of strings: spellsspells = ["protego", "accio", "expecto patronum", "legilimens"]​# Use map() to apply a lambda function over spells: shout_spellsshout_spells = map(lambda item: item + '!!!', spells)​# Convert shout_spells to a list: shout_spells_listshout_spells_list = list(shout_spells)​# Convert shout_spells into a list and print itprint(shout_spells_list)​<script.py> output:    ['protego!!!', 'accio!!!', 'expecto patronum!!!', 'legilimens!!!']
```

**Filter\(\) and lambda functions**

```text
 # Create a list of strings: fellowshipfellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']​# Use filter() to apply a lambda function over fellowship: resultresult = filter(lambda member: len(member) > 6, fellowship)​# Convert result to a list: result_listresult_list = list(result)​# Convert result into a list and print itprint(result_list)​<script.py> output:    ['samwise', 'aragorn', 'legolas', 'boromir']
```

#### Iterators

**Iterables**

* Examples: Lists, strings, dictionaries, file connections
* An _object_ with an associated `iter()` method
* Apply `iter()` to an _iterable_ creates an _iterator_, an object with an associated `next()` method

```text
 it = iter('Da')next(it)'D'next(it)'a'
```

**Writing an iterator to load data in chunks**

```text
 import pandas as pdimport matplotlib.pyplot as plt​# Initialize reader object: urb_pop_readerurb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)​# Initialize empty DataFrame: datadata = pd.DataFrame()​# Iterate over each DataFrame chunkfor df_urb_pop in urb_pop_reader:​    # Check out specific country: df_pop_ceb    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']​    # Zip DataFrame columns of interest: pops    pops = zip(df_pop_ceb['Total Population'],                df_pop_ceb['Urban population (% of total)'])​    # Turn zip object into list: pops_list    pops_list = list(pops)​    # Use list comprehension to create new DataFrame column 'Total Urban Population'    df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1]) for tup in pops_list]​    # Append DataFrame chunk to data: data    data = data.append(df_pop_ceb)​# Plot urban population datadata.plot(kind='scatter', x='Year', y='Total Urban Population')plt.show()
```

### Broadcasting

```text
 # Assigning scaler value (`np.nan` in this case) to column slice broadcasts value to each row.# The slice consists of every 3rd row starting from 0 in the last column.APPL.iL=loc[::3, -1] = np.nan
```

### Joining DataFrames

Which should you use? Use the simplest tool that works.

* df1.append\(df2\) : stacking _vertically_
* pd.concat\(\[df1, df2\]\):
  * stacking many horizontally or vertically
  * simple inner/outer joins on indexes
* df1.join\(df2\): inner/outer/_left_/_right_ joins on indexes
* pd.merge\(\[df1, df2\]\): many joins on multiple _columns_

## NLP

### Data Cleaning and Text Preprocessing

Removing HTML Markup: The BeautifulSoup Package

[https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)

#### Tokenization

[https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words):

Splitting the documents into individual words.

```text
 lower_case = letters_only.lower()        # Convert to lower casewords = lower_case.split()               # Split into words
```

#### Stop Words

[https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words):

> Finally, we need to decide how to deal with frequently occurring words that don't carry much meaning. Such words are called "stop words"; in English they include words such as "a", "and", "is", and "the". Conveniently, there are Python packages that come with stop word lists built in. Let's import a stop word list from the Python Natural Language Toolkit \(NLTK\). You'll need to install the library if you don't already have it on your computer; you'll also need to install the data packages that come with it, as follows:

```text
 import nltknltk.download()  # Download text data sets, including stop wordsfrom nltk.corpus import stopwords # Import the stop word listprint stopwords.words("english")# Remove stop words from "words"words = [w for w in words if not w in stopwords.words("english")]print words
```

#### unicode string

[https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words):

> Don't worry about the "u" before each word; it just indicates that Python is internally representing each word as a unicode string.

#### other

> There are many other things we could do to the data - For example, Porter Stemming and Lemmatizing \(both available in NLTK\) would allow us to treat "messages", "message", and "messaging" as the same word, which could certainly be useful. However, for simplicity, the tutorial will stop here.

### Extracting features from text files

#### Bag of Words Model

According to [scikit-learn](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#extracting-features-from-text-files):

> In order to perform machine learning on text documents, we first need to turn the text content into numerical feature vectors.

> The most intuitive way to do so is the bags of words representation...

According to [Wikipedia](https://www.wikiwand.com/en/Bag-of-words_model):

> The bag-of-words model is a simplifying representation used in natural language processing and information retrieval \(IR\). In this model, a text \(such as a sentence or a document\) is represented as the bag \(multiset\) of its words, disregarding grammar and even word order but keeping multiplicity.

It is a _simplifying_ representation as grammar and word order are disregarded.

[https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)

> Now that we have our training reviews tidied up, how do we convert them to some kind of numeric representation for machine learning? One common approach is called a Bag of Words. The Bag of Words model learns a vocabulary from all of the documents, then models each document by counting the number of times each word appears. For example, consider the following two sentences:

> Sentence 1: "The cat sat on the hat"

> Sentence 2: "The dog ate the cat and the hat"

> From these two sentences, our vocabulary is as follows:

> { the, cat, sat, on, hat, dog, ate, and }

> To get our bags of words, we count the number of times each word occurs in each sentence. In Sentence 1, "the" appears twice, and "cat", "sat", "on", and "hat" each appear once, so the feature vector for Sentence 1 is:

> { the, cat, sat, on, hat, dog, ate, and }

> Sentence 1: { 2, 1, 1, 1, 1, 0, 0, 0 }

> Similarly, the features for Sentence 2 are: { 3, 1, 0, 0, 1, 1, 1, 1}

> In the IMDB data, we have a very large number of reviews, which will give us a large vocabulary. To limit the size of the feature vectors, we should choose some maximum vocabulary size. Below, we use the 5000 most frequent words \(remembering that stop words have already been removed\).

```text
 print "Creating the bag of words...\n"from sklearn.feature_extraction.text import CountVectorizer​# Initialize the "CountVectorizer" object, which is scikit-learn's# bag of words tool.  vectorizer = CountVectorizer(analyzer = "word",   \                             tokenizer = None,    \                             preprocessor = None, \                             stop_words = None,   \                             max_features = 5000)​# fit_transform() does two functions: First, it fits the model# and learns the vocabulary; second, it transforms our training data# into feature vectors. The input to fit_transform should be a list of# strings.train_data_features = vectorizer.fit_transform(clean_train_reviews)​# Numpy arrays are easy to work with, so convert the result to an# arraytrain_data_features = train_data_features.toarray()​>>> print train_data_features.shape(25000, 5000)
```

> It has 25,000 rows and 5,000 features \(one for each vocabulary word\).

> Note that CountVectorizer comes with its own options to automatically do preprocessing, tokenization, and stop word removal -- for each of these, instead of specifying "None", we could have used a built-in method or specified our own function to use. See [the function documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) for more details. However, we wanted to write our own function for data cleaning in this tutorial to show you how it's done step by step.

> Now that the Bag of Words model is trained, let's look at the vocabulary:

```text
 # Take a look at the words in the vocabularyvocab = vectorizer.get_feature_names()print vocab
```

> If you're interested, you can also print the counts of each word in the vocabulary:

```text
 import numpy as np​# Sum up the counts of each vocabulary worddist = np.sum(train_data_features, axis=0)​# For each, print the vocabulary word and the number of times it# appears in the training setfor tag, count in zip(vocab, dist):    print count, tag
```

Bag of words is a model of text data that can be converted to and feed to machining learning algorithm for model induction.

#### TF-IDF

From [http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/](http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/):

TF-IDF stands for "Term Frequency, Inverse Document Frequency". It is a way to score the importance of words \(or "terms"\) in a document based on how frequently they appear across multiple documents.

Intuitively...If a word appears frequently in a document, it's important. Give the word a high score.But if a word appears in many documents, it's not a unique identifier. Give the word a low score.Therefore, common words like "the" and "for", which appear in many documents, will be scaled down. Words that appear frequently in a single document will be scaled up.

[http://scikit-learn.org/stable/tutorial/text\_analytics/working\_with\_text\_data.html](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

According to [scikit-learn](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#extracting-features-from-text-files):

> Occurrence count is a good start but there is an issue: longer documents will have higher average count values than shorter documents, even though they might talk about the same topics.

> To avoid these potential discrepancies it suffices to divide the number of occurrences of each word in a document by the total number of words in the document: these new features are called `tf` for Term Frequencies.

> Another refinement on top of `tf` is to downscale weights for words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus.

> This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.

```text
 >>> print("\n".join(twenty_train.data[0].split("\n")[:3]))From: sd345@city.ac.uk (Michael Collier)Subject: Converting images to HP LaserJet III?Nntp-Posting-Host: hampton​>>> from sklearn.feature_extraction.text import CountVectorizer>>> count_vect = CountVectorizer()>>> X_train_counts = count_vect.fit_transform(twenty_train.data)>>> X_train_counts.shape(2257, 35788)
```

`TfidfTransformer` can be used to transform the count-matrix to a tf-idf representation:

```text
 >>> tfidf_transformer = TfidfTransformer()>>> X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)>>> X_train_tfidf.shape(2257, 35788)
```

### Training a classifier

> Now that we have our features, we can train a classifier to try to predict the category of a post. Let’s start with a naïve Bayes classifier, which provides a nice baseline for this task. `scikit-learn` includes several variants of this classifier; the one most suitable for word counts is the multinomial variant:

```text
 >>> from sklearn.naive_bayes import MultinomialNB>>> clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)​>>> docs_new = ['God is love', 'OpenGL on the GPU is fast']>>> X_new_counts = count_vect.transform(docs_new)>>> X_new_tfidf = tfidf_transformer.transform(X_new_counts)​>>> predicted = clf.predict(X_new_tfidf)​>>> for doc, category in zip(docs_new, predicted):...     print('%r => %s' % (doc, twenty_train.target_names[category]))...'God is love' => soc.religion.christian'OpenGL on the GPU is fast' => comp.graphics
```

### Building a pipeline

> In order to make the vectorizer =&gt; transformer =&gt; classifier easier to work with, scikit-learn provides a Pipeline class that behaves like a compound classifier:

```text
 >>> from sklearn.pipeline import Pipeline>>> text_clf = Pipeline([('vect', CountVectorizer()),...                      ('tfidf', TfidfTransformer()),...                      ('clf', MultinomialNB()),... ])
```

The names `vect`, `tfidf` and `clf` \(classifier\) are arbitrary. We shall see their use in the section on grid search, below. We can now train the model with a single command:

```text
 >>> text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
```

### Evaluation of the performance on the test set

> Evaluating the predictive accuracy of the model is equally easy:

```text
 >>> import numpy as np>>> twenty_test = fetch_20newsgroups(subset='test',...     categories=categories, shuffle=True, random_state=42)>>> docs_test = twenty_test.data>>> predicted = text_clf.predict(docs_test)>>> np.mean(predicted == twenty_test.target)            0.834...
```

> I.e., we achieved 83.4% accuracy. Let’s see if we can do better with a linear support vector machine \(SVM\), which is widely regarded as one of the best text classification algorithms \(although it’s also a bit slower than naïve Bayes\). We can change the learner by just plugging a different classifier object into our pipeline:

```text
 >>> from sklearn.linear_model import SGDClassifier>>> text_clf = Pipeline([('vect', CountVectorizer()),...                      ('tfidf', TfidfTransformer()),...                      ('clf', SGDClassifier(loss='hinge', penalty='l2',...                                            alpha=1e-3, n_iter=5, random_state=42)),... ])>>> _ = text_clf.fit(twenty_train.data, twenty_train.target)>>> predicted = text_clf.predict(docs_test)>>> np.mean(predicted == twenty_test.target)            0.912...
```

> `scikit-learn` further provides utilities for more detailed performance analysis of the results:

```text
 >>> from sklearn import metrics>>> print(metrics.classification_report(twenty_test.target, predicted,...     target_names=twenty_test.target_names))...                                                                 precision    recall  f1-score   support​           alt.atheism       0.95      0.81      0.87       319         comp.graphics       0.88      0.97      0.92       389               sci.med       0.94      0.90      0.92       396soc.religion.christian       0.90      0.95      0.93       398​           avg / total       0.92      0.91      0.91      1502​​>>> metrics.confusion_matrix(twenty_test.target, predicted)array([[258,  11,  15,  35],       [  4, 379,   3,   3],       [  5,  33, 355,   3],       [  5,  10,   4, 379]])
```

> As expected the confusion matrix shows that posts from the newsgroups on atheism and christian are more often confused for one another than with computer graphics.

### Parameter tuning using grid search

[http://scikit-learn.org/stable/tutorial/text\_analytics/working\_with\_text\_data.html\#from-occurrences-to-frequencies](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#from-occurrences-to-frequencies)

### Additional Reference

[http://scikit-learn.org/stable/modules/feature\_extraction.html\#text-feature-extraction](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

## Industry News

* [中銀與騰訊攜手成立金融科技聯合實驗室](http://www.bank-of-china.com/big5/aboutboc/bi1/201706/t20170622_9651860.html)
* [農行與百度戰略合作 攜手成立金融科技聯合實驗室](https://www.jiqizhixin.com/articles/ba76cf39-e9d5-494d-b292-464f95170c5b)

### Deep learning

* image classification
* face recognition
* self driving car

## Career

[80000 Hours](https://80000hours.org/career-reviews/machine-learning-phd/):

> According to someone we spoke to in the industry, a machine learning PhD is a good preparation for getting a high earning job in a quantitative hedge fund.

> Machine learning is a hot area a lot of people want to get into, so there is a risk that it could become more difficult to get jobs as lots of people crowd into the area. For example, MIT’s introduction to machine learning course recently had 700 people sign up – they had to use an overflow lecture room and deliberately weed people out of the course early.&lt;sup&gt;[18](https://80000hours.org/career-reviews/machine-learning-phd/#fn-18)&lt;/sup&gt; If machine learning turns out to progress more slowly than is expected, and doesn’t live up to the hype, then the number of jobs might shrink as well.

> If you’re going for machine learning research positions in academia or industry, you’ll need a PhD. It can also be helpful to have a PhD \(or even further academic experience\) to set up a startup that develops cutting-edge techniques. But for most non-research positions in industry, including at top firms such as Google, a master’s is sufficient. For industry jobs that aren’t pure research, going on to get a PhD isn’t an advantage and probably isn’t worth the time cost.

> Also bear in mind that many data-analysis problems that companies and nonprofits have don’t require machine learning to solve – often good solutions can be built using simple data science methods instead.&lt;sub&gt;[19](https://80000hours.org/career-reviews/machine-learning-phd/#fn-19)&lt;/sub&gt;

### How to become machine learning researcher

[80000 Hours](https://80000hours.org/career-reviews/machine-learning-phd/):

> Read research papers and try replicating their results \(Andrew Ng [recommends](https://youtu.be/F1ka6a13S9I?t=1h10m25s) this to become an excellent machine learning researcher\). Some papers you could do this with:

> [Important deep learning papers](https://github.com/ChristosChristofidis/awesome-deep-learning)[Important reinforcement learning papers](https://github.com/aikorea/awesome-rl#papers--thesis)Papers from [top conferences](https://80000hours.org/ai-safety-syllabus/#conferences)

### Other advice

> Software Engineering — For Type A data science, let me be clear: engineering is a separate discipline. So if this is the type of data scientist you want to become, you do not need to be an engineer. However, if you want to put machine learning algorithms into production \(i.e. Type B\), you will need a strong foundation in software engineering.[How to Become a Data Scientist \(Part 1/3\)](https://medium.com/towards-data-science/how-to-become-a-data-scientist-part-1-3-8706a62b809e)

### How to get a job

#### Independent projects

Posting your code is extremely important. In fact, having a Github repository posted online is a powerful signal that you are a competent data scientist \(it is a competence trigger, which we will discuss in a moment\).

Kaggle is the simplest way to complete independent projects, but there are many other ways. There are three parts to completing an independent data science project:

1. Coming up with an idea
2. Acquiring the data
3. Analyzing the data and/or building a model

Kaggle is great, because steps 1 and 2 are completed for you. But a huge amount of data science is exactly those parts, so Kaggle can’t fully prepare you for a job as a data scientist. I will help you now with steps 1 and 2 by giving you a list of a few ideas for independent data science projects. I encourage you to steal these.

1. Use Latent Semantic Analysis to extract topics from tweets. Pull the data using the Twitter API.
2. Use a bag of words model to cluster the top questions on /r/AskReddit. Pull the data using the Reddit API.
3. Identify interesting traffic volume spikes for certain Wikipedia pages and correlate them to news events. Access and analyze the data by using AWS Open Datasets and Amazon Elastic MapReduce.
4. Find topic networks in Wikipedia by examining the link graph in Wikipedia. Use another AWS Open Datasets.I mention a few other sample projects in Becoming a Data Hacker.

#### Competence triggers

I absolutely think you should have a Github page, and you should post code from your independent projects there. If you have performed decently well in a couple of Kaggle competitions, then your Kaggle profile will be impressive, too. Answering questions on StackExchange or Quora can be a bit of a distraction from your real work, so it should not be a priority. And starting your own blog is great, but probably not necessary. As an alternative to a blog, you can focus on writing good documentation in a README in your Github repositories.

### CV

For example, a story that works for me is, “I am an experienced data scientist, I have a great math background, and I am good at explaining complicated stuff.” If I were applying to a more software development-focused data science job, a possible story for me could be, “I have experience building really fast and accurate machine-learning models in Python. I also understand big data technology like Hadoop.” For a more business-focused role, a story could be, “I have experience using stats and machine-learning to find useful insights in data. I also have experience presenting those insights with dashboards and automated reports, and I am good at public speaking.” When you come up with your story, don’t be afraid to try some different ones on for size. All of the three stories I just wrote are true about me. It’s all about positioning yourself the right way for the company.[http://will-stanton.com/creating-a-great-data-science-resume](http://will-stanton.com/creating-a-great-data-science-resume)/

Because your resume is there to tell a targeted story in order to get an interview, you really should not have any skills or technologies listed that do not fit with that story. For example, if your story is all about being a “PhD in Computer Science with deep understanding of neural networks and the ability to explain technical topics,” you probably should not include your experience with WordPress. Including general skills like HTML and CSS is probably good, but you probably do not need to list that you are an expert in Knockout.JS and elastiCSS. This advice is doubly true for non-technical skills like “customer service” or “phone direct sales.” Including things like that actually makes the rest of your resume look worse, because it emphasizes that you have been focused on a lot of things other than data science, and — worse — that you do not really understand what the team is looking for. If you want to include something like that to add color to your resume, you should add it in the “Additional Info” section at the end of the resume, not in the “Skills and Technologies” section.[http://will-stanton.com/creating-a-great-data-science-resume](http://will-stanton.com/creating-a-great-data-science-resume)/

### Interview Questions

#### Why data science is science?

> Before we delve any deeper, it is worth taking a moment to reflect on the ‘science’ in ‘data science’, because — in a sense — all scientists are data scientists, as they all work with data in one form or another. But to take what is generally considered to be data science in industry, what actually makes it a science? Great question! The answer should be: ‘the scientific method’. Given the multi-disciplinary nature of science, the scientific method is the one thing that binds the fields together. If you got this right, full marks to you.[How to Become a Data Scientist \(Part 1/3\)](https://medium.com/towards-data-science/how-to-become-a-data-scientist-part-1-3-8706a62b809e)

#### Why you're suitable for this job

**Strong problem solving ability**

> Clearly, you need to possess the tools to solve the problems, but they are just that: tools. In this sense, even the statistical/machine learning techniques can be thought of as the tools by which you solve problems. New techniques arise, technology evolves; the one constant is problem solving.[How to Become a Data Scientist \(Part 1/3\)](https://medium.com/towards-data-science/how-to-become-a-data-scientist-part-1-3-8706a62b809e)

#### My strength comparatively

Communication / Business Acumen:Had opportunities to communicate with many different people internally and externally and at different levels

> Having the ability to conceptualise business problems and the environment in which they occur is critical. And translating statistical insights into recommended actions and implications to a lay audience is absolutely crucial, particularly for Type A data science. I was chatting to Yanir about this, and this is how he put it:“I find it weird how some technical people don’t pay attention to how non-technical people’s eyes glaze over when they start using jargon. It’s really important to put yourself in the listener’s/reader’s shoes”[How to Become a Data Scientist \(Part 1/3\)](https://medium.com/towards-data-science/how-to-become-a-data-scientist-part-1-3-8706a62b809e)

Have hacker mindset \(and show the portfolio\)

## Terms I don't understand yet

From an article named [When Not to Use Deep Learning](http://hyperparameter.space/blog/when-not-to-use-deep-learning/):

* strong prior
* representation
* embedding
* one-shot learning
* GAN
* stochastic gradient descent
* Bayesian inference
* Markov chain
* variational approximation to the posterior
* optimizers

## Bookmark

[Seeing Theory](http://students.brown.edu/seeing-theory/) A very beautiful visual introduction to probability and statistics.

## Fun facts

Sam Walton, founder of Walmart, in the 1950s used airplanes to fly over and count cars on parking lots to assess real estate investments.JP Morgan - Big Data and AI Approach to Investment Management

## challenges of big data and machine learning

> Potential Pitfalls of Big Data and Machine Learning: The transition to a Big Data framework will not be without setbacks. Certain types of data may lead into blind alleys - datasets that don’t contain alpha, signals that have too little investment capacity, decay quickly, or are simply too expensive to purchase. Managers may invest too much into unnecessary infrastructure e.g. build complex models and architecture that don’t justify marginal performance improvements. Machine Learning algorithms cannot entirely replace human intuition. Sophisticalted models, if not properly guided, can overfit or uncover spurious relationships and patterns. Talent will present another source of risk – employing data scientists who lack specific financial expertise or financial intuition may not lead to the desired investment results or lead to culture clashes. In implementing Big Data and Machine Learning in finance, it is more important to understand the economics behind data and signals, than to be able to develop complex technological solutions. Many Big Data and AI concepts may sound plausible but will not lead to viable trading strategies.

JP Morgan - Big Data and AI Approach to Investment Management

## 为什么AI现阶段不能完全替代金融学的各种模型？

A. 无法很好的用AI来定义一个金融问题现阶段比较被商业化广泛应用的机器学习还是监督学习，而监督学习要求有明确的问题定义。现在看起来很有希望的强化学习，迁移学习等还并不能大规模普及应用。以简单的监督学习为例，如果你想建立一个模型来预测企业并购是否会影响公司股价，那么你需要提供大量并购数据，以及并购后股价是否发生了变动。理想情况下，在收集足够多的并购消息和股价变动信息后，做自然语言分析后提取特征放到机器学习模型里面就大功告成了。然而在实际情况中：我们无法给出明确的问题定义和边界。如果想用AI来来制定一个股票交易策略，那么需要考虑进去多少因素？仅仅只考虑并购消息就够了么？越多的相关的因素越可以提高模型的拟合性和准确性。如宏观政策和微观的具体情况都会影响到股价的波动，漏掉其中哪一个都会造成一定的影响，往往是多多益善。在这种情况下，每个问题都需要大量人和数据来支撑，这也是为什么大量用AI来预测股票走势的探索都无疾而终的原因。现阶段或者可预见的未来，在很多问题上不会出现这种明确的定义和范围。

B. AI从业者和金融从业者缺乏有效沟通在很长的时间里面，计算机和金融学之间的联系相对比较薄弱。作为一个CS背景的人，我个人对于金融/经济学的理解还处于比较肤浅的状态，只理解基本的概念和原理。同样的，金融服务类从业者又缺乏对于AI模型和统计的了解。因此使用AI来推动金融学发展需要大量跨领域的人才，至少需要两个方向都懂的项目经理。

C. 金融领域缺乏足够的大数据和人工智能人才储备人工智能的火爆，或者说06年Hinton论文后带起的深度学习的老树开花，并没有来得及为行业储存大量的专业人才。不难看出，大量一流AI/ML人才还是被互联网公司一网打尽，\(Hinton在谷歌Lecun 在FB\)留给金融服务类公司的人才并不多。以我们公司举例，各国分公司的Chief Data Scientist 基本都不是计算机/统计/数学背景出身的科学家。

D. 投出产出在现阶段不成正比，短时间内难以获得收益。在这种情况下，每个问题都需要大量人和数据来支撑。因此研究探索型的、不能产生利润的方向很少有公司来投资AI来进行研究的。换言之，有财力提供AI研究的金融公司不多，小型的金融机构或者学术机构又缺乏资源（资金，技术人才，数据积累）来进行相关系统的研究。

E. 技术性的难题还包括很多，比如AI在金融领域应该以什么样的模式存在？是一个软件，一个网络服务，还是一个机器人。在大量需要与客户沟通的领域，人机交互以及如何生成内容也是继续探索的领域。

