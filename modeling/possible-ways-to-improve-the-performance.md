# Possible Ways to improve the performance

[2nd place ](http://blog.kaggle.com/2016/03/17/airbnb-new-user-bookings-winners-interview-2nd-place-keiichi-kuroyanagi-keiku/)of a Kaggle competition:

> When I made several attempts to build it, I found that some features decreased the NDCG@5 score, so I selected randomly features at the ratio of 90% and built repeatedly a single XGBoost many times. Finally, I selected the best XGBoost model \(5 fold-CV: 0.833714\) from the built models

