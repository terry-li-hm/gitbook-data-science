# Notes



{% embed data="{\"url\":\"https://twitter.com/TessFerrandez/status/953334000311029766\",\"type\":\"rich\",\"title\":\"Tess on Twitter\",\"description\":\"Takeaways from @AndrewYNg talk - AI is the new electricity https://t.co/z1ePKKFP6M \#sketchnotes \#MachineLearning pic.twitter.com/GqexGr5JvN— Tess \(@TessFerrandez\) January 16, 2018\n\n\",\"icon\":{\"type\":\"icon\",\"url\":\"https://abs.twimg.com/icons/apple-touch-icon-192x192.png\",\"width\":192,\"height\":192,\"aspectRatio\":1},\"embed\":{\"type\":\"app\",\"html\":\"<blockquote class=\\"twitter-tweet\\" align=\\"center\\"><p lang=\\"en\\" dir=\\"ltr\\">Takeaways from <a href=\\"https://twitter.com/AndrewYNg?ref\_src=twsrc%5Etfw\\">@AndrewYNg</a> talk - AI is the new electricity <a href=\\"https://t.co/z1ePKKFP6M\\">https://t.co/z1ePKKFP6M</a> <a href=\\"https://twitter.com/hashtag/sketchnotes?src=hash&amp;ref\_src=twsrc%5Etfw\\">\#sketchnotes</a> <a href=\\"https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref\_src=twsrc%5Etfw\\">\#MachineLearning</a> <a href=\\"https://t.co/GqexGr5JvN\\">pic.twitter.com/GqexGr5JvN</a></p>&mdash; Tess \(@TessFerrandez\) <a href=\\"https://twitter.com/TessFerrandez/status/953334000311029766?ref\_src=twsrc%5Etfw\\">January 16, 2018</a></blockquote>\n<script async src=\\"https://platform.twitter.com/widgets.js\\" charset=\\"utf-8\\"></script>\n\",\"maxWidth\":550,\"aspectRatio\":1}}" %}

{% embed data="{\"url\":\"https://twitter.com/mbostock/status/991517711250305024\",\"type\":\"rich\",\"title\":\"Mike Bostock on Twitter\",\"description\":\"Don’t compare percentage change on a linear scale; use a log scale instead. -50% \(0.5×\) is as big a change as +100% \(2×\). pic.twitter.com/EhjtTG0d2M— Mike Bostock \(@mbostock\) May 2, 2018\n\n\",\"icon\":{\"type\":\"icon\",\"url\":\"https://abs.twimg.com/icons/apple-touch-icon-192x192.png\",\"width\":192,\"height\":192,\"aspectRatio\":1},\"embed\":{\"type\":\"app\",\"html\":\"<blockquote class=\\"twitter-tweet\\" align=\\"center\\"><p lang=\\"en\\" dir=\\"ltr\\">Don’t compare percentage change on a linear scale; use a log scale instead. -50% \(0.5×\) is as big a change as +100% \(2×\). <a href=\\"https://t.co/EhjtTG0d2M\\">pic.twitter.com/EhjtTG0d2M</a></p>&mdash; Mike Bostock \(@mbostock\) <a href=\\"https://twitter.com/mbostock/status/991517711250305024?ref\_src=twsrc%5Etfw\\">May 2, 2018</a></blockquote>\n<script async src=\\"https://platform.twitter.com/widgets.js\\" charset=\\"utf-8\\"></script>\n\",\"maxWidth\":550,\"aspectRatio\":1}}" %}

  


Root Mean Squared Error \(RMSE\)The square root of the mean/average of the square of all of the error.  
The use of RMSE is very common and it makes an excellent general purpose error metric for numerical predictions.  
Compared to the similar Mean Absolute Error, RMSE amplifies and severely punishes large errors.  
[https://www.kaggle.com/wiki/RootMeanSquaredError](https://www.kaggle.com/wiki/RootMeanSquaredError)  
from sklearn.metrics import mean\_squared\_errorRMSE = mean\_squared\_error\(y, y\_pred\)\*\*0.5  


You may ask why should I care about gradient boosting when machine learning seems to be all about deep learning? The answer is that it works very well for structured data.XGBoost has become so successful with the Kaggle data science community, to the point of [“winning practically every competition in the structured data category”](https://www.import.io/post/how-to-win-a-kaggle-competition/).

Essential to these successes is the use of “LSTMs,” a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version. Almost all exciting results based on recurrent neural networks are achieved with them. It’s these LSTMs that this essay will explore.

Since computation graph in PyTorch is defined at runtime you can use tour favorite Python debugging tools such as pdb, ipdb, PyCharm debugger or old trusty print statements.

In practice regular RNNs are rarely used anymore, while GRUs and LSTMs dominate the field.

By the way, one of the absolute best books I’ve read on this topic \(and neural nets/deep learning in general\) is the just released [Hands-On Machine Learning with Scikit-Learn and Tensorflow](http://amzn.to/2p5t4Ll). I saw a few earlier editions and they really upped my game. Don’t wait, just grab it ASAP. It rocks. It goes into a ton more detail than I have here but I’ll give you the basics to get you moving in the right direction fast.

Batch Normalization helps your network learn faster by “smoothing” the values at various stages in the stack. Exactly why this works is seemingly not well-understood yet, but it has the effect of helping your network converge much faster, meaning it achieves higher accuracy with less training, or higher accuracy after the same amount of training, often dramatically so.



Precision is the percentage of relevant items out of those that have been returned, while recall is the percentage of relevant items that have been returned out of the overall number of relevant items. Hence, it is easy to artificially increase recall to 100% by always returning all the items in the database, but this would mean settling for near-zero precision. Similarly, one can increase precision by always returning a single item that the algorithm is very confident about, but this means that recall would suffer. Ultimately, the best balance between precision and recall depends on the application.



A word embedding is an approach to provide a dense vector representation of words that capture something about their meaning.

Word embeddings are an improvement over simpler bag-of-word model word encoding schemes like word counts and frequencies that result in large and sparse vectors \(mostly 0 values\) that describe documents but not the meaning of the words.

Word embeddings work by using an algorithm to train a set of fixed-length dense and continuous-valued vectors based on a large corpus of text. Each word is represented by a point in the embedding space and these points are learned and moved around based on the words that surround the target word.

It is defining a word by the company that it keeps that allows the word embedding to learn something about the meaning of words. The vector space representation of the words provides a projection where words with similar meanings are locally clustered within the space.

The use of word embeddings over other text representations is one of the key methods that has led to breakthrough performance with deep neural networks on problems like machine translation.



Word2vec is one algorithm for learning a word embedding from a text corpus.

There are two main training algorithms that can be used to learn the embedding from text; they are continuous bag of words \(CBOW\) and skip grams.

We will not get into the algorithms other than to say that they generally look at a window of words for each target word to provide context and in turn meaning for words. The approach was developed by Tomas Mikolov, formerly at Google and currently at Facebook.



"If you are looking for a career where your services will be in high demand, you should find something where you provide a scarce, complementary service to something that is getting ubiquitous and cheap. So what’s getting ubiquitous and cheap? Data. And what is complementary to data? Analysis"— Hal Varian, UC Berkeley, Chief Economist at Google



A random forest is a bunch of independent decision trees each contributing a “vote” to an prediction. E.g. if there are 50 trees, and 32 say “rainy” and 18 say “sunny”, then the score for “rainy” is 32/50, or 64,% and the score for a “sunny” is 18/50, or 36%. Since 64% &gt; 36%, the forest has voted that they think it will rain.  
  
When you add more decision trees to a random forest, they decide what they think INDEPENDENTLY of all the other trees. They learn on their own, and when it comes time to make a prediction, they all just throw their own uninfluenced opinion into the pot.  
  
A gradient boosting model is a CHAIN of decision trees that also each make a vote. But instead of each learning in isolation, when you add a new one to the chain, it tries to improve a bit on what the rest of the chain already thinks. So, a new tree’s decision IS influenced by all the trees that have already voiced an opinion.  
  
Unlike a Random Forest, when you add a new tree to a GBM, it gets to see what its predecessors thought - and how they got it right or wrong. They then formulate a suggestion to correct the errors of their predecessors - and then they add that to the pot, and then the process continues with the next tree you add to the chain.



Gradient boosting  
Gradient boosting is a type of boosting. It relies on the intuition that the best possible next model, when combined with previous models, minimizes the overall prediction error. The key idea is to set the target outcomes for this next model in order to minimize the error. How are the targets calculated? The target outcome for each case in the data depends on how much changing that case’s prediction impacts the overall prediction error:  
  
If a small change in the prediction for a case causes a large drop in error, then next target outcome of the case is a high value. Predictions from the new model that are close to its targets will reduce the error.  
If a small change in the prediction for a case causes no change in error, then next target outcome of the case is zero. Changing this prediction does not decrease the error.  
The name gradient boosting arises because target outcomes for each case are set based on the gradient of the error with respect to the prediction. Each new model takes a step in the direction that minimizes prediction error, in the space of possible predictions for each training case.



Ensembles and boosting  
Machine learning models can be fitted to data individually, or combined in an ensemble. An ensemble is a combination of simple individual models that together create a more powerful new model.  
  
Boosting is a method for creating an ensemble. It starts by fitting an initial model \(e.g. a tree or linear regression\) to the data. Then a second model is built that focuses on accurately predicting the cases where the first model performs poorly. The combination of these two models is expected to be better than either model alone. Repeat the process many times. Each successive model attempts to correct for the shortcomings of the combined boosted ensemble of all previous models.





## Data Science for Business

### 1. Introduction: Data-Analytic Thinking

### 2. Business Problems and Data Science Solutions

### 3. Introduction to Predictive Modeling: From Coorelation to Supervised Segmentation

### 4. Fitting a Model to Data

### 5. Overfitting and Its Avoidance

### 6. Similiarity, Neighbors, and Clusters

### 7. Decision Analytic Thinkings I: What Is a Good Model?

### 8. Visualizing Model Performance

Common visualizations:

1. ROC curve
2. Cumulative response curve: tp rate \(tp divided by totally number of positives\) \(y axis\) vs. percentage of the population that is targeted \(x axis\)
3. Lift curve

> One of the reaons accuracy is a poor metric is that it is misleading when daasets are skews…

\(e.g. 93% negatives and 7% positives.\) AUC is a better metric.

> Even modest AUC scores may lead to good business results.

> A critical part of the data scientist's job is aranging for proper evaluation of models and conveying this information to stakeholders. Doing this well takes expereince, but it is vital in order to reduce superises and to manage expectations among all concerned. Visualizatino of reults is an important piece of the evaluation task.
>
> When building a model from data, adjusting the training samplein various ways may be useful or even necessary; but evluation should use a sample reflecting the original, realistic population so that the resutls reflect waht will actually be achieved.
>
> When the costs and benefits of decisions can be specified, the data scientist can calculate an expected cost per instance for each modeland simply choose whicever model produces the best value. In some cases a basic profit graph can be useful to compare modesl of interest under a range of conditions. These graphs may be easy to comprehend for stakeholders who are not data scientists, since they reduce model performance to their basic "bottom line" cost or profit.
>
> The disavantage of a profit graph is that it requires that operating conditions be known and specified exactly. With many real-world problems, the operating conditions are imprecise or change over time, and the data scientist must contend with uncertainty. In such cases other graphs may be more useful. When costs and benefits cannot be specified with confidence, but the class mix will likely not change, a _cumulative response_ or _lift_ graph is useful. Both show the relative advantages of classifiers, independent of the value \(monetary or otherwise\) of the advantages.
>
> Finally, ROC curves are a valuable visualization tool for the data scientist. Though they take some practive to interpret readily, they seperate out performance from operating conditions. In doing so they convey the fundametal trade-offs that each model is making.



Basic flow per the readings so far:

Step1: Exploratory Data Analysis

Step2: Data Preprocessing

Step 3: Feature Engineering

Step 4: Model Selection and Training

Step 5: Model Ensemble

Some tips

设置random seed，使得你的模型reproduce，以Random Foreset举例：

seed=0clf=RandomForestClassifier\(random\_state=seed\)

每个project组织好文件层次和布局，既方便与其他人交流，也方便自己。比如在一个project下，分设3个文件夹，一个是input，放训练数据、测试数据，一个model，放模型文件，最后一个submission文件，放你生成要提交的结果文件。具体的可以参考[这里](https://www.kaggle.com/wiki/ModelSubmissionBestPractices)

From [http://www.jianshu.com/p/32def2294ae6](http://www.jianshu.com/p/32def2294ae6)





## Data Science Notes

### Machine Learning Process

![](.gitbook/assets/image%20%2823%29.png)

Unsupervised learning and supervised learning can be used together. ![](file:///Users/Terry/Downloads/Data%20Science/DS%20Notes/Images/unsupervised%20and%20supervised.png?lastModify=1525080918)

![](.gitbook/assets/image%20%2818%29.png)

### Metric

#### F1

> My identifier has a really great F1. This is the best of both worlds. Both my false positive and false negative rates are low, which means that I can identify POI's reliable and accurately. If my identifier finds a POI then the person is almost certainly a POI, and if the identifier does not flag someone, then they are almost certainly not a POI.

#### Precision and Recall

> My identifier doesn't have great **precision**, but it does have good **recall**. That means that, nearly every time a POI shows up in my test set, I am able to identify him or her. The cost of this is that I sometime get some false positives, where non-POIs get flagged.
>
> My identifier doesn't have great **recall**, but it does have good **precision**. That means that whenever a POI gets flagged in my test set, I know with a lot of confidence that it's very likely to be a real POI and not a false alarm. On the other hand, the price I pay for this is that I sometimes miss real POIs, since I'm effectively reluctant to pull the trigger on edge cases.

### kNN

#### Disadvantage

Linear regression can extrapolate the ends but kNN can't:

![](.gitbook/assets/image%20%2813%29.png)



## Quora Kaggle Competition

Questions

* For data balancing, normally the objective is to make the ratio of positive and negative classes be 50/50. But in the [kernel by anokas](https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb) he made the ratio of the training set from 37% to 17%, which is the ratio of the test set. There is a [relevant discussion thread](https://www.kaggle.com/c/quora-question-pairs/discussion/31179) initiated by sweezjeezy.
* According to [YantingCao](https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb#181191), the re-sampling approach in the [kernel by anokas](https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb) leak information into the cross validation set. In other words, the cross validation set contains data from the training set. anokas also agreed on this. My questions are:
  * What does 're-sampling' mean?
  * Why such approach lead to such issue?

Notes

Since our metric is log loss, resampling the data to represent the same distribution \(of 0.165\) will give us a much better score in Public LB. The ratio of the training set can be observed directly. The ratio of the test set can be calculated using the result of a [naive submission which use the ratio of the training set as the estimated probability](https://www.kaggle.io/svf/1077333/f8eecce4cf447dccad546c8ec882e0d1/__results__.html#Test-Submission) and [a bit of magic algebra](https://www.kaggle.com/davidthaler/quora-question-pairs/how-many-1-s-are-in-the-public-lb) as there is only one distribution of classes that could have produced this score. It seems from the discussion that such method is only applicable to evaluation with logloss function.

Suggested by [Paul Larmuseau](https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb#179520):

> I wonder you could beat your splendid results if you correct one error... since you use .lower\(\).split\(\) you don't split the '?' from the last word, i presume since i discover that one of your least frequent words has a '?' fulfill?'. So i would 'clean' the data with a replace '?' with ' ' or i prefer to use nltk.word\_tokenize\(q1\) , so doing this you would split all words from the punctuations, and make disappear one type of error, as in that case you throw away nearly all last words of all questions as 'rare' words, or you cluster together all questions with the same last word. Since i didnot install yet that wonderfull XGaboost, i give you the honour to improve or worsen \(i don't know yet\) your own results.

According to [Philipp Schmidt](https://www.kaggle.io/svf/1082541/22b67ba5fac3793c4b56697f1d0906e8/__results__.html#Feature-construction), to limit the computational complexity and storage requirements we can subsample the training set by `dfs = df[0:2500]`. After subsampling we should check if the distribution of the positive and negative classes of the subsample is similar to the original training set. If not we need to find out a better sampling method.

#### Text Analysis using Machine Learning

Most of the algorithms accept only numerical feature vectors \(`vector` is a one dimensional `array` in computer science\). So we need to convert the text documents into numerical features vectors with a fixed size in order to make use of the machining learning algorithms for text analysis.

This can be done by the following steps:

1. Assign each of the words in the text documents an integer ID. Each of the words is called a `token`. This step is called `tokenization`.
2. Count the occurrences of tokens in each document. This step is called `counting`. The count of each token is created as a feature.
3. `Normalization` \(**Don't understand what it means at this moment**\)

**\(to add easy-to-understand example\)**

This process is called `vectorization`. The resulting numerical feature vectors is called a `bag-of-words` representation.

One issue of `vectorization` is that longer documents will have higher average count values than shorter documents while they might talk about the same topic. The solution is to divide the number of occurrences of each word in a document by total number of words in the document. These features are called `term frequency` or `tf`.

Another issue `vectorization` is that in a large text corpus the common words like "the", "a", "is" will shadow the rare words during the model induction. The solution is to downscale the weight of the words that appear in many documents. This downscaling is called `term frequency times inverse document frequency` or `tf-idf` .

I learnt the above from a [scikit-learn tutorial](http://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation).

According to [Kaggle](https://www.kaggle.com/c/quora-question-pairs/rules), `word embedding` is an example of `pre-trained models`. The followings are the embeddings mentioned by [Kaggle competitors](https://www.kaggle.com/c/quora-question-pairs/discussion/30286):

* [word2vec by Google](https://code.google.com/archive/p/word2vec/)
* [GloVe](https://nlp.stanford.edu/projects/glove/)
* [fastText by Facebook](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

[Kaggle](https://www.kaggle.com/c/quora-question-pairs/discussion/30286) requires competitors to share the pre-trained models and word embeddings used to "keep the competition fair by making sure that everyone has access to the same data and pretrained models."

What is `pre-trained models`?

What is `word embedding`?

Some other tools:

* [Gensim](https://radimrehurek.com/gensim/)
* [spaCy](https://spacy.io/)
* [Amazon Machine Learning](https://aws.amazon.com/machine-learning/)



## Gradient Descent

{% embed data="{\"url\":\"https://www.dropbox.com/s/ekee45cl4tel2x2/sgd\_bad.gif?dl=0\",\"type\":\"link\",\"title\":\"sgd\_bad.gif\",\"description\":\"Shared with Dropbox\",\"icon\":{\"type\":\"icon\",\"url\":\"https://cfl.dropboxstatic.com/static/images/logo\_catalog/dropbox\_webclip\_152\_m1-vflU0bwfQ.png\",\"width\":152,\"height\":152,\"aspectRatio\":1},\"thumbnail\":{\"type\":\"thumbnail\",\"url\":\"https://www.dropbox.com/temp\_thumb\_from\_token/s/ekee45cl4tel2x2?preserve\_transparency=False&size=1024x1024&size\_mode=2\",\"width\":1024,\"height\":1024,\"aspectRatio\":1}}" %}

{% embed data="{\"url\":\"https://www.dropbox.com/s/5ptildzwybdtjp3/sgd.gif?dl=0\",\"type\":\"link\",\"title\":\"sgd.gif\",\"description\":\"Shared with Dropbox\",\"icon\":{\"type\":\"icon\",\"url\":\"https://cfl.dropboxstatic.com/static/images/logo\_catalog/dropbox\_webclip\_152\_m1-vflU0bwfQ.png\",\"width\":152,\"height\":152,\"aspectRatio\":1},\"thumbnail\":{\"type\":\"thumbnail\",\"url\":\"https://www.dropbox.com/temp\_thumb\_from\_token/s/5ptildzwybdtjp3?preserve\_transparency=False&size=1024x1024&size\_mode=2\",\"width\":1024,\"height\":1024,\"aspectRatio\":1}}" %}

* If the learning rate is too large \(0.01\), the cost may oscillate up and down. It may even diverge \(refer to the graph above\).
* A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.
* In deep learning, we usually recommend that you:
* Choose the learning rate that better minimizes the cost function.
* If your model overfits, use other techniques to reduce overfitting.
* Complexity of machine learning comes from the data rather than from the lines of code.
* Intuitions from one domain or from one application area often do not transfer to other application areas.
* Applied deep learning is a very iterative process where you just have to go around this cycle many times to hopefully find a good choice of network for your application.

## Tips

* Preprocessing the dataset is important.
* You implemented each function separately: initialize\(\), propagate\(\), optimize\(\). Then you built a model\(\).
* The larger models \(with more hidden units\) are able to fit the training set better, until eventually the largest models overfit the data.

## General methodology to build a Neural Network:

1. Define the neural network structure \( \# of input units, \# of hidden units, etc\).
2. Initialize the model's parameters
3. Loop:
   * Implement forward propagation
   * Compute loss
   * Implement backward propagation to get the gradients
   * Update parameters \(gradient descent\)

## Initialization

* Random initialization is used to break symmetry and make sure different hidden units can learn different things.
  * The weights ​$W^{\[l\]}​$ should be initialized randomly to break symmetry.
  * It is however okay to initialize the biases $b^{\[l\]}​$​ to zeros. Symmetry is still broken so long as ​$W^{\[l\]}​$ is initialized randomly.
* Initializing weights to very large random values does not work well.
* Hopefully initializing with small random values does better. The important question is: how small should be these random values be? Lets find out in the next part!
* Different initializations lead to different results
* He initialization works well for networks with ReLU activations.

## Deal with high variance

### Regularization

* Regularization will help you reduce overfitting.
* Regularization will drive your weights to lower values.
* L2 regularization and Dropout are two very effective regularization techniques.

#### L2-regularization

* What is L2-regularization actually doing? L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.
* the implications of L2-regularization on:
  * The cost computation: A regularization term is added to the cost
  * The backpropagation function: There are extra terms in the gradients with respect to weight matrices
  * Weights end up smaller \("weight decay"\): Weights are pushed to smaller values.

#### Dropout

* Dropout is a regularization technique.
* You only use dropout during training. Don't use dropout \(randomly eliminate nodes\) during test time.
* Apply dropout both during forward and backward propagation.
* During training time, divide each dropout layer by keep\_prob to keep the same expected value for the activations. For example, if keep\_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep\_prob is other values than 0.5.

## Gradient Checking

* Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient \(computed using forward propagation\).
* Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.

## Mini-batch Gradient Descent and Stochastic Gradient Descent \(SGD\)

* The difference between gradient descent, mini-batch gradient descent and stochastic gradient descent is the number of examples you use to perform one update step.
* You have to tune a learning rate hyperparameter ​.
* With a well-turned mini-batch size, usually it outperforms either gradient descent or stochastic gradient descent \(particularly when the training set is large\).
* Shuffling and Partitioning are the two steps required to build mini-batches
* Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.

## Gradient descent with Momentum

* Common values for ​ range from 0.8 to 0.999. If you don't feel inclined to tune this, ​$\beta = 0.9​$ is often a reasonable default.
* Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
* You have to tune a momentum hyperparameter ​$\beta​$ and a learning rate ​$\alpha​$.





* 楊強教授表示，目前國內的 AI 創業公司主要分為兩種，一種傾向於只關心技術，而另一種則有具體的場景和數據，這也是他個人更加看好的。「人工智能目前要完全顛覆一個領域的可能性還是比較渺茫，比較成功的案例還是集中於漸進的模式。也就是說，人工智能作為整個產業鏈條的一部分而存在，但並不會取代後者。因此，數據需要不斷循環更新，而場景則提供了數據更新的環境。」
* 楊強提到，AI的成功有著5大必要條件：
  * 清晰的商業模式
  * 高質量的大數據
  * 清晰的問題定義和領域邊間
  * 懂人工智能的跨界人才，擅長應用和算法
  * 計算能力



## Windows

Remote access Jupyter notebook from Windows

> 1. Download the latest version of [PUTTY](http://www.putty.org/)
> 2. Open PUTTY and enter the server URL or IP address as the hostname
> 3. Now, go to SSH on the bottom of the left pane to expand the menu and then click on Tunnels
> 4. Enter the port number which you want to use to access Jupyter on your local machine. Choose 8000 or greater \(ie 8001, 8002, etc.\) to avoid ports used by other services, and set the destination as localhost:8888 where :8888 is the number of the port that Jupyter Notebook is running on. Now click the Add button, and the ports should appear in the Forwarded ports list.
> 5. Finally, click the Open button to connect to the server via SSH and tunnel the desired ports. Navigate to [http://localhost:8000](http://localhost:8000/) \(or whatever port you chose\) in a web browser to connect to Jupyter Notebook running on the server.





