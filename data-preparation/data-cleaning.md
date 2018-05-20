# Data Cleaning

[Best Python Data Cleaning Libraries For People Data - Junjay's Blog](http://junjaytan.com/blog/python-data-cleaning-people-contact-data/)

[Handy Python Libraries for Formatting and Cleaning Data](https://blog.modeanalytics.com/python-data-cleaning-libraries/)

[A programmer’s cleaning guide for messy sensor data \| Opensource.com](https://opensource.com/article/17/9/messy-sensor-data)

[Simple approach to handle missing values \| Kaggle](https://www.kaggle.com/kostya17/simple-approach-to-handle-missing-values)

[vi3k6i5/flashtext: Extract Keywords from sentence or Replace keywords in sentences.](https://github.com/vi3k6i5/flashtext)

[Fixing Typos \| Kaggle](https://www.kaggle.com/steubk/fixing-typos)

### Imputation {#2)-A-Better-Option:-Imputation}

> Imputation fills in the missing value with some number. The imputed value won't be exactly right in most cases, but it usually gives more accurate models than dropping the column entirely.
>
> This is done with
>
> ```python
> from sklearn.preprocessing import Imputer
> my_imputer = Imputer()
> data_with_imputed_values = my_imputer.fit_transform(original_data)
> ```
>
> The default behavior fills in the mean value for imputation. Statisticians have researched more complex strategies, but those complex strategies typically give no benefit once you plug the results into sophisticated machine learning models.
>
> One \(of many\) nice things about Imputation is that it can be included in a scikit-learn Pipeline. Pipelines simplify model building, model validation and model deployment.

#### An Extension To Imputation

> Imputation is the standard approach, and it usually works well. However, imputed values may by systematically above or below their actual values \(which weren't collected in the dataset\). Or rows with missing values may be unique in some other way. In that case, your model would make better predictions by considering which values were originally missing. 
>
> ```python
> imputed_X_train_plus = X_train.copy()
> imputed_X_test_plus = X_test.copy()
>
> cols_with_missing = (col for col in X_train.columns 
>                                  if X_train[col].isnull().any())
> for col in cols_with_missing:
>     imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
>     imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
>
> # Imputation
> my_imputer = Imputer()
> imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
> imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)
> ```
>
> In some cases this approach will meaningfully improve results. In other cases, it doesn't help at all.

![](../.gitbook/assets/image%20%281%29.png)

![](../.gitbook/assets/image%20%2821%29.png)

![](../.gitbook/assets/image%20%2830%29.png)

