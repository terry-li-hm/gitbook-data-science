# Machine Learning Yearning

{% embed data="{\"url\":\"https://www.dropbox.com/s/98cxhqe4uvdjz51/Ng\_MLY01.pdf?dl=0\",\"type\":\"link\",\"title\":\"Ng\_MLY01.pdf\",\"description\":\"Shared with Dropbox\",\"icon\":{\"type\":\"icon\",\"url\":\"https://cfl.dropboxstatic.com/static/images/logo\_catalog/dropbox\_webclip\_152\_m1-vflU0bwfQ.png\",\"width\":152,\"height\":152,\"aspectRatio\":1},\"thumbnail\":{\"type\":\"thumbnail\",\"url\":\"https://cfl.dropboxstatic.com/static/images/logo\_catalog/twitter-card-glyph\_m1@2x-vflrCAzPX.png\",\"width\":320,\"height\":320,\"aspectRatio\":1}}" %}

{% embed data="{\"url\":\"https://www.dropbox.com/s/jowspahz8w6j6w2/Ng\_MLY02.pdf?dl=0\",\"type\":\"link\",\"title\":\"Ng\_MLY02.pdf\",\"description\":\"Shared with Dropbox\",\"icon\":{\"type\":\"icon\",\"url\":\"https://cfl.dropboxstatic.com/static/images/logo\_catalog/dropbox\_webclip\_152\_m1-vflU0bwfQ.png\",\"width\":152,\"height\":152,\"aspectRatio\":1},\"thumbnail\":{\"type\":\"thumbnail\",\"url\":\"https://cfl.dropboxstatic.com/static/images/logo\_catalog/twitter-card-glyph\_m1@2x-vflrCAzPX.png\",\"width\":320,\"height\":320,\"aspectRatio\":1}}" %}

{% embed data="{\"url\":\"https://www.dropbox.com/s/u9y9j64htoz79gi/Ng\_MLY03.pdf?dl=0\",\"type\":\"link\",\"title\":\"Ng\_MLY03.pdf\",\"description\":\"Shared with Dropbox\",\"icon\":{\"type\":\"icon\",\"url\":\"https://cfl.dropboxstatic.com/static/images/logo\_catalog/dropbox\_webclip\_152\_m1-vflU0bwfQ.png\",\"width\":152,\"height\":152,\"aspectRatio\":1},\"thumbnail\":{\"type\":\"thumbnail\",\"url\":\"https://cfl.dropboxstatic.com/static/images/logo\_catalog/twitter-card-glyph\_m1@2x-vflrCAzPX.png\",\"width\":320,\"height\":320,\"aspectRatio\":1}}" %}





> This diagram shows NNs doing better in the regime of small datasets. This effect is less consistent than the effect of NNs doing well in the regime of huge datasets. In the small data regime, depending on how the features are hand-engineered, traditional algorithms may or may not do better. For example, if you have 20 training examples, it might not matter much whether you use logistic regression or a neural network; the hand-engineering of features will have a bigger effect than the choice of algorithm. But if you have 1 million examples, I would favor the neural network.

## Setting up development and test sets

> But if the dev and test sets come from different distributions, then your options are less clear. Several things could have gone wrong:
>
> 1. You had overfit to the dev set.
> 2. The test set is harder than the dev set. So your algorithm might be doing as well as could    be expected, and there’s no further significant improvement is possible.
> 3. The test set is not necessarily harder, but just different, from the dev set. So what works well on the dev set just does not work well on the test set. In this case, a lot of your work    to improve dev set performance might be wasted effort.

> If you are trading off N different criteria, such as binary file size of the model \(which is important for mobile apps, since users don’t want to download large apps\), running time, and accuracy, you might consider setting N-1 of the criteria as “satisficing” metrics. I.e., you simply require that they meet a certain value. Then define the final one as the “optimizing” metric. For example, set a threshold for what is acceptable for binary file size and running time, and try to optimize accuracy given those constraints.

> If you later realize that your initial dev/test set or metric missed the mark, by all means change them quickly. For example, if your dev set + metric ranks classifier A above classifier B, but your team thinks that classifier B is actually superior for your product, then this might be a sign that you need to change your dev/test sets or your evaluation metric.
>
> There are three main possible causes of the dev set/metric incorrectly rating classifier A higher:
>
> 1. The actual distribution you need to do well on is different from the dev/test sets.
> 2. You have overfit to the dev set.  
>    If you need to track your team’s progress, you can also evaluate your system regularly—say
>
>    once per week or once per month—on the test set. But do not use the test set to make any
>
>    decisions regarding the algorithm, including whether to roll back to the previous week’s
>
>    system. If you do so, you will start to overfit to the test set, and can no longer count on it to
>
>    give a completely unbiased estimate of your system’s performance \(which you would need if
>
>    you’re publishing research papers, or perhaps using this metric to make important business
>
>    decisions\).
>
> 3. The metric is measuring something other than what the project needs to optimize.

#### Takeaways: Setting up development and test sets

> * Choose dev and test sets from a distribution that reflects what data you expect to get in   the future and want to do well on. This may not be the same as your training data’s   distribution.
> * Choose dev and test sets from the same distribution if possible.
> * Choose a single-number evaluation metric for your team to optimize. If there are multiple   goals that you care about, consider combining them into a single formula \(such as   averaging multiple error metrics\) or defining satisficing and optimizing metrics.
> * Machine learning is a highly iterative process: You may try many dozens of ideas before   finding one that you’re satisfied with. 
> * Having dev/test sets and a single-number evaluation metric helps you quickly evaluate   algorithms, and therefore iterate faster.
> * When starting out on a brand new application, try to establish dev/test sets and a metric   quickly, say in less than a week. It might be okay to take longer on mature applications.
> * The old heuristic of a 70%/30% train/test split does not apply for problems where you   have a lot of data; the dev and test sets can be much less than 30% of the data.
> * Your dev set should be large enough to detect meaningful changes in the accuracy of your   algorithm, but not necessarily much larger. Your test set should be big enough to give you   a confident estimate of the final performance of your system.
> * If your dev set and metric are no longer pointing your team in the right direction, quickly   change them: \(i\) If you had overfit the dev set, get more dev set data. \(ii\) If the actual   distribution you care about is different from the dev/test set distribution, get new   dev/test set data. \(iii\) If your metric is no longer measuring what is most important to   you, change the metric.

#### Examples of what can be tried to improve the model

> * Get more data: Collect more pictures of cats.
> * Collect a more diverse training set. For example, pictures of cats in unusual positions; cats   with unusual coloration; pictures shot with a variety of camera settings; ….
> * Train the algorithm longer, by running more gradient descent iterations.
> * Try a bigger neural network, with more layers/hidden units/parameters.
> * Try a smaller neural network.
> * Try adding regularization \(such as L2 regularization\).
> * Change the neural network architecture \(activation function, number of hidden units, etc.\)

> If you choose well among these possible directions, you’ll build the leading cat picture platform, and lead your company to success. If you choose poorly, you might waste months.

## Basic Error Analysis

> * When you start a new project, especially if it is in an area in which you are not an expert,  it is hard to correctly guess the most promising directions.
> * So don’t start off trying to design and build the perfect system. Instead build and train a   basic system as quickly as possible—perhaps in a few days. Then use error analysis to   help you identify the most promising directions and iteratively improve your algorithm   from there.
> * Carry out error analysis by manually examining ~100 dev set examples the algorithm   misclassifies and counting the major categories of errors. Use this information to   prioritize what types of errors to work on fixing.
> * Consider splitting the dev set into an Eyeball dev set, which you will manually examine,   and a Blackbox dev set, which you will not manually examine. If performance on the   Eyeball dev set is much better than the Blackbox dev set, you have overfit the Eyeball dev   set and should consider acquiring more data for it.
> * The Eyeball dev set should be big enough so that your algorithm misclassifies enough   examples for you to analyze. A Blackbox dev set of 1,000-10,000 examples is sufficient   for many applications.
> * If your dev set is not big enough to split this way, just use an Eyeball dev set for manual   error analysis, model selection, and hyperparameter tuning.



