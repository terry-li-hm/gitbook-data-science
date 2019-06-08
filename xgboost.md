# XGBoost

From [XGBoost Tutorials](https://xgboost.readthedocs.io/en/latest/tutorials/index.html):

> For common cases such as ads clickthrough log, the dataset is extremely imbalanced. This can affect the training of XGBoost model, and there are two ways to improve it.
>
> * If you care only about the overall performance metric \(AUC\) of your prediction
>   * Balance the positive and negative weights via `scale_pos_weight`
>   * Use AUC for evaluation
> * If you care about predicting the right probability
>   * In such a case, you cannot re-balance the dataset
>   * Set parameter `max_delta_step` to a finite number \(say 1\) to help convergence



From [XGBoost Tutorials](https://xgboost.readthedocs.io/en/latest/tutorials/index.html):

> When you observe high training accuracy, but low test accuracy, it is likely that you encountered overfitting problem.
>
> There are in general two ways that you can control overfitting in XGBoost:
>
> * The first way is to directly control model complexity.
>   * This includes `max_depth`, `min_child_weight` and `gamma`.
> * The second way is to add randomness to make training robust to noise.
>   * This includes `subsample` and `colsample_bytree`.
>   * You can also reduce stepsize `eta`. Remember to increase `num_round` when you do so.



A good plot from a [Medium post](https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e):

![](.gitbook/assets/image%20%288%29.png)





* [FYI - For those of you using the xgboostExplainer package](https://www.reddit.com/r/learnmachinelearning/comments/9n2kq2/fyi_for_those_of_you_using_the_xgboostexplainer/)

## Explanation

* [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
* [pred\_contribs = True](https://xgboost.readthedocs.io/en/latest/python/python_api.html)

