# Categorical Variables

> There is some debate about the relative merits of these approaches, and some models can deal with label encoded categorical variables with no issues. [Here is a good Stack Overflow discussion](https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor). I think \(and this is just a personal opinion\) for categorical variables with many classes, one-hot encoding is the safest approach because it does not impose arbitrary values to categories. The only downside to one-hot encoding is that the number of features \(dimensions of the data\) can explode with categorical variables with many categories. To deal with this, we can perform one-hot encoding followed by [PCA](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf) or other [dimensionality reduction methods](https://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/) to reduce the number of dimensions \(while still trying to preserve information\).



Per [a MOOC](https://www.coursera.org/lecture/competitive-data-science/categorical-and-ordinal-features-qu1TF):

* Label and frequency encodings are often used for tree-based models
* One-hot encoding is often used for non-tree-based models \(e.g. kNN, nerual networks\)

This is a [good summary ](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63)of the common strategies. But not sure if they are really helpful as Jeremy didn't talk about them.



## Change the data type from Object to Category

```python
categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]

categorical_feats
for f_ in categorical_feats:
    data[f_], _ = pd.factorize(data[f_])
    # Set feature type as categorical
    data[f_] = data[f_].astype('category')
```

```python
cols_to_exclude = ['Program_Year', 'Date_of_Payment', 'Payment_Publication_Date']
for col in df.columns:
    if df[col].nunique() < 600 and col not in cols_to_exclude:
        df[col] = df[col].astype('category')
```

Olivier Grellier, Senior Data Scientist at H2O.ai, [does so](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances).

### Benefit

> * We can define a custom sort order which can improve summarizing and reporting the data. In the example above, “X-Small” &lt; “Small” &lt; “Medium” &lt; “Large” &lt; “X-Large”. Alphabetical sorting would not be able to reproduce that order.
> * Some of the python visualization libraries can interpret the categorical data type to apply approrpiate statistical models or plot types.
> * Categorical data uses less memory which can lead to performance improvements.

But make sure you define all the possible categories, otherwise any value you didn't define will become `NaN`. Search for "Let’s build" in this [article ](https://pbpython.com/pandas_dtypes_cat.html)for details.

### Set the order

Per [Jeremy](https://youtu.be/CzdWqFTmn0Y?t=55m40s), it is not very important to do so but good to do so.

```python
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'],
    ordered=True, inplace=True)
```

