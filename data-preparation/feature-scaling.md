# Feature Scaling

According to [this great tutorial ](https://www.kaggle.com/colinmorris/embedding-layers)from Kaggle, scaling features and target variable help achieve better results, at least for deep learning.

> **Aside**: I'm passing in `df.y` as my target variable rather than `df.rating`. The `y` column is just a 'centered' version of the rating - i.e. the rating column minus its mean over the training set. For example, if the overall average rating in the training set was 3 stars, then we would translate 3 star ratings to 0, 5 star ratings to 2.0, etc. to get `y`. This is a common practice in deep learning, and tends to help achieve better results in fewer epochs. For more details, feel free to check out [this kernel](https://www.kaggle.com/colinmorris/movielens-preprocessing) with all the preprocessing I performed on the MovieLens dataset.

