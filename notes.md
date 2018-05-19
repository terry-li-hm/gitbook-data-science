# Notes

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





