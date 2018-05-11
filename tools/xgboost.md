# XGBoost

[Complete Guide to Parameter Tuning in XGBoost \(with codes in Python\)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)



> 1. Reserve a portion of training set as the validation set.
> 2. Set `eta` to a relatively high value \(e.g. 0.05 ~ 0.1\), `num_round` to 300 ~ 500.
> 3. Use grid search to find the best combination of other parameters.
> 4. Gradually lower `eta` until we reach the optimum.
> 5. **Use the validation set as `watch_list` to re-train the model with the best parameters. Observe how score changes on validation set in each iteration. Find the optimal value for `early_stopping_rounds`.**

