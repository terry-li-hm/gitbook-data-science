# Data Preparation

* [How \(dis\)similar are my train and test data? – Towards Data Science](https://towardsdatascience.com/how-dis-similar-are-my-train-and-test-data-56af3923de9b)
* [Google AI Blog: Preprocessing for Machine Learning with tf.Transform](https://ai.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html)
* [Using Python to Figure out Sample Sizes for your Study – Mark Nagelberg](http://www.marknagelberg.com/using-python-to-figure-out-sample-sizes-for-your-study/)
* [Splitting into train, dev and test sets](https://cs230-stanford.github.io/train-dev-test-split.html)
* [Comprehensive Guide to 12 Dimensionality Reduction Techniques](https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/)
* [great-expectations/great\_expectations: Always know what to expect from your data.](https://github.com/great-expectations/great_expectations)

## Setting up development and test sets \(From Machine Learning Yearning\)

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

