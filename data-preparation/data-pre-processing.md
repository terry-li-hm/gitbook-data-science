# Data Pre-Processing

## Function to Convert Data Types <a id="Function-to-Convert-Data-Types"></a>

This will help reduce memory usage by using more efficient types for the variables. For example `category` is often a better type than `object` \(unless the number of unique categories is close to the number of rows in the dataframe\).

```python
import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df
```





```python
# Function to calculate missing values by column# Funct 
def missing_values_table(df, print_info = False):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        if print_info:
            # Print some summary information
            print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
```





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

