# Data Processing

* [Pandas Concatenation Tutorial](https://www.dataquest.io/blog/pandas-concatenation-tutorial/)
* [Pendulum - Python datetimes made easy](https://pendulum.eustace.io/)
* [Common Excel Tasks Demonstrated in Pandas - Practical Business Python](http://pbpython.com/excel-pandas-comp.html)
* [Pandas Pivot Table Explained - Practical Business Python](http://pbpython.com/pandas-pivot-table-explained.html)
* [Overview of Pandas Data Types - Practical Business Python](http://pbpython.com/pandas_dtypes.html)
* [Understanding the Transform Function in Pandas - Practical Business Python](http://pbpython.com/pandas_transform.html)
* [Guide to Encoding Categorical Values in Python - Practical Business Python](http://pbpython.com/categorical-encoding.html)
* [How to One Hot Encode Sequence Data in Python](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)
* [Video series: Easier data analysis in Python using the pandas library](http://www.dataschool.io/easier-data-analysis-with-pandas/)
* [WZBSocialScienceCenter/pdftabextract: A set of tools for extracting tables from PDF files helping to do data mining on \(OCR-processed\) scanned documents.](https://github.com/WZBSocialScienceCenter/pdftabextract)
* [Data Pre-Processing in Python: How I learned to love parallelized applies with Dask and Numba](https://towardsdatascience.com/how-i-learned-to-love-parallelized-applies-with-python-pandas-dask-and-numba-f06b0b367138)



Numpy array can only contain one data type.

In a simplified sense, you can think of series as a 1D labelled array.

* [Introduction to Numpy -1 : An absolute beginners guide to Machine Learning and Data science.](https://hackernoon.com/introduction-to-numpy-1-an-absolute-beginners-guide-to-machine-learning-and-data-science-5d87f13f0d51)
* [How to Index, Slice and Reshape NumPy Arrays for Machine Learning in Python](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)
* [Intro to Pandas: -1 : An absolute beginners guide to Machine Learning and Data science.](https://hackernoon.com/intro-to-pandas-1-an-absolute-beginners-guide-to-machine-learning-and-data-science-a1fed3a6f0f3)
* [Pandas tips and tricks – Towards Data Science](https://towardsdatascience.com/pandas-tips-and-tricks-33bcc8a40bb9)

### Select Data

Data can be selected from Pandas DataFrame using:

* Square brackets
* the `loc` method
  * label-based \(i.e. using the labels of columns and observations\)
  * inclusive
* the `iloc` method
  * position-based \(i.e. using the index of columns and observations\)
  * exclusive

Examples:

```python
# Print out country column as Pandas Series
print(cars['country'])​

# Print out country column as Pandas DataFrame
print(cars[['country']]) 

# Print out DataFrame with country and drives_right columns
print(cars[['country','drives_right']])

​# Print out first 3 observations
print(cars[:3]) ​

# Print out fourth, fifth and sixth observation
print(cars[3:6])

​# Print out drives_right column as Series
print(cars.loc[:, 'drives_right'])​

# Print out drives_right column as DataFrame
print(cars.loc[:,['drives_right']])​

# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:,['cars_per_cap','drives_right']])
```

The `loc` and `iloc` method are more powerful but square brackets are good enough for selecting all observations of some columns.

### 

## Example of some Pandas techniques

```python
import pandas as pd
path = '~/.kaggle/competitions/msk-redefining-cancer-treatment/'

df_labels_test = pd.read_csv(path + 'stage1_solution_filtered.csv')
df_labels_test.head(2)
```

|  | ID | class1 | class2 | class3 | class4 | class5 | class6 | class7 | class8 | class9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 19 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

```python
df_labels_test['Class'] = pd.to_numeric(df_labels_test.drop('ID', axis=1).idxmax(axis=1).str[5:])
df_labels_test.head(2)
```

|  | ID | class1 | class2 | class3 | class4 | class5 | class6 | class7 | class8 | class9 | Class |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 1 | 19 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |

## 



> It is important to discover and quantify the degree to which variables in your dataset are dependent upon each other. This knowledge can help you better prepare your data to meet the expectations of machine learning algorithms, such as linear regression, whose performance will degrade with the presence of these interdependencies.

> The performance of some algorithms can deteriorate if two or more variables are tightly related, called multicollinearity. An example is linear regression, where one of the offending correlated variables should be removed in order to improve the skill of the model.



## Re-sampling

Since our metric is log loss, resampling the data to represent the same distribution \(of 0.165\) will give us a much better score in Public LB. The ratio of the training set can be observed directly. The ratio of the test set can be calculated using the result of a [naive submission which use the ratio of the training set as the estimated probability](https://www.kaggle.io/svf/1077333/f8eecce4cf447dccad546c8ec882e0d1/__results__.html#Test-Submission) and [a bit of magic algebra](https://www.kaggle.com/davidthaler/quora-question-pairs/how-many-1-s-are-in-the-public-lb) as there is only one distribution of classes that could have produced this score. It seems from the discussion that such method is only applicable to evaluation with logloss function.

## Dask

* [python︱大规模数据存储与读取、并行计算：Dask库简述 - CSDN博客](https://blog.csdn.net/sinat_26917383/article/details/78044437)
* [【干货】Dask快速搭建分布式集群（大数据0基础可以理解，并使用！） - CSDN博客](https://blog.csdn.net/a19990412/article/details/79510219)
* [Python visualization with datashader](https://yeshuanova.github.io/blog/posts/python-visualization-datashader/)
* [Data Pre-Processing in Python: How I learned to love parallelized applies with Dask and Numba](https://towardsdatascience.com/how-i-learned-to-love-parallelized-applies-with-python-pandas-dask-and-numba-f06b0b367138)

