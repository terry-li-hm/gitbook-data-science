# Imbalanced Class

* [Deep learning unbalanced training data?Solve it like this.](https://medium.com/@shub777_56374/deep-learning-unbalanced-training-data-solve-it-like-this-6c528e9efea6)
* [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/index.html)



## Use F1 or precision and recall

According to [Alvira Swalin](https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428) \(student of Jeremy\):

> When you have a small positive class, then F1 score makes more sense. This is the common problem in fraud detection where positive labels are few.

According to another [Medium post](https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba):

> **Use precision and recall to focus on small positive class —** When the positive class is smaller and the ability to detect correctly positive samples is our main focus \(correct detection of negatives examples is less important to the problem\) we should use precision and recall.



From [XGBoost Tutorials](https://xgboost.readthedocs.io/en/latest/tutorials/index.html):

> For common cases such as ads clickthrough log, the dataset is extremely imbalanced. This can affect the training of XGBoost model, and there are two ways to improve it.
>
> * If you care only about the overall performance metric \(AUC\) of your prediction
>   * Balance the positive and negative weights via `scale_pos_weight`
>   * Use AUC for evaluation
> * If you care about predicting the right probability
>   * In such a case, you cannot re-balance the dataset
>   * Set parameter `max_delta_step` to a finite number \(say 1\) to help convergence

