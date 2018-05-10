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

### 

