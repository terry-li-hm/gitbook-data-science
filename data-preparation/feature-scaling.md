# Feature Scaling

According to [this great tutorial ](https://www.kaggle.com/colinmorris/embedding-layers)from Kaggle, scaling features and target variable help achieve better results, at least for deep learning.

> **Aside**: I'm passing in `df.y` as my target variable rather than `df.rating`. The `y` column is just a 'centered' version of the rating - i.e. the rating column minus its mean over the training set. For example, if the overall average rating in the training set was 3 stars, then we would translate 3 star ratings to 0, 5 star ratings to 2.0, etc. to get `y`. This is a common practice in deep learning, and tends to help achieve better results in fewer epochs. For more details, feel free to check out [this kernel](https://www.kaggle.com/colinmorris/movielens-preprocessing) with all the preprocessing I performed on the MovieLens dataset.

Algorithm that needs scaling:

> Rule of thumb I follow here is any algorithm that computes distance or assumes normality, **scale your features!!!**
>
> Some examples of algorithms where feature scaling matters are:
>
> * **k-nearest neighbors** with an Euclidean distance measure is sensitive to magnitudes and hence should be scaled for all features to weigh in equally.
> * Scaling is critical, while performing **Principal Component Analysis\(PCA\)**. PCA tries to get the features with maximum variance and the variance is high for high magnitude features. This skews the PCA towards high magnitude features.
> * We can speed up **gradient descent** by scaling. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

Algorithm that doesn't need scaling:

> * **Tree based models** are not distance based models and can handle varying ranges of features. Hence, Scaling is not required while modelling trees.
> * Algorithms like **Linear Discriminant Analysis\(LDA\), Naive Bayes** are by design equipped to handle this and gives weights to the features accordingly. Performing a features scaling in these algorithms may not have much effect.

