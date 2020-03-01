# Data Pre-Processing

```python
def remove_missing_columns(train, test, threshold = 90):
    # Calculate missing stats for train and test (remember to calculate a percent!)
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(train)
    
    test_miss = pd.DataFrame(test.isnull().sum())
    test_miss['percent'] = 100 * test_miss[0] / len(test)
    
    # list of missing columns for train and test
    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])
    
    # Combine the two lists together
    missing_columns = list(set(missing_train_columns + missing_test_columns))
    
    # Print information
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))
    
    # Drop the missing columns and return
    train = train.drop(columns = missing_columns)
    test = test.drop(columns = missing_columns)
    
    return train, test    
```



## One-hot Encoding

> 一、要不要one-hot？
>
> 这在机器学习界也有争论。理论上，树模型如果够深，也能将关键的类别型特型切出来。
>
> 关于这个，xgboost的作者tqchen在某个[issues](https://github.com/dmlc/xgboost/issues/95)有提到过：
>
> I do not know what you mean by vector. xgboost treat every input feature as numerical, with support for missing values and sparsity. The decision is at the user
>
> So if you want ordered variables, you can transform the variables into numerical levels\(say age\). Or if you prefer treat it as categorical variable, do one hot encoding.
>
> 在另一个[issues](https://github.com/szilard/benchm-ml/issues/1)上也提到过（tqchen commented on 8 May 2015）：
>
> One-hot encoding could be helpful when the number of categories are small\( in level of 10 to 100\). In such case one-hot encoding can discover interesting interactions like \(gender=male\) AND \(job = teacher\).
>
> While ordering them makes it harder to be discovered\(need two split on job\). However, indeed there is not a unified way handling categorical features in trees, and usually what tree was really good at was ordered continuous features anyway..
>
> 总结起来的结论，大至两条：
>
> * 1.对于类别有序的类别型变量，比如age等，当成数值型变量处理可以的。对于非类别有序的类别型变量，推荐one-hot。但是one-hot会增加内存开销以及训练时间开销。
> * 2.类别型变量在范围较小时（tqchen给出的是\[10,100\]范围内）推荐使用

