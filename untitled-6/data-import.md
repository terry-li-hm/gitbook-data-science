# Data import

* [openpyxl / openpyxl — Bitbucket](https://bitbucket.org/openpyxl/openpyxl/src/default/)
* [python-excel/xlrd: Library for developers to extract data from Microsoft Excel \(tm\) spreadsheet files](https://github.com/python-excel/xlrd/)
* [Read Untidy Excel Files • tidyxl](https://nacnudus.github.io/tidyxl/)
* [The Best Tools to Analyze Alternative Data \| Parts 2 & 3: Ingesting and Loading Data - AlternativeData](https://alternativedata.org/the-best-tools-to-analyze-alternative-data-parts-2-3-ingesting-and-loading-data/)
* [TensorFlow全新的数据读取方式：Dataset API入门教程 \| 雷锋网](https://www.leiphone.com/news/201711/zV7yM5W1dFrzs8W5.html)



For large data set, Jeremy would [set the data type with minimum number of bits ](https://youtu.be/YSFG_W8JxBo?t=1130)for `int` and `float`. For example:

```python
types = {'id': 'int64',
         'item_nbr': 'int32',
         'store_nbr': 'int8',
         'unit_sales': 'float32',
         'onpromotion': 'object'}
```

## Data type for Boolean

For a Boolean field with missing value, you can only set it with data type `object` \(which is slow and memory heavy\). So what you can do is read it as `object` first, do some analysis to see how to impute it, and convert it into data type `Boolean` after the imputation. Here's an example code by Jeremy:

```python
df_all.onpromotion.fillna(False, inplace=True)
df_all.onpromotion = df_all.onpromotion.map({'False': False, 
                                             'True': True})
df_all.onpromotion = df_all.onpromotion.astype(bool)
```



