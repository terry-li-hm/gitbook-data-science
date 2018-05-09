# Python

It is a good practice to create separate .py for a class and call it, instead of doing these in the same .py 



```python
# NULL or missing values check:
print("Nulls in Oil columns: {0} => {1}".format(oil.columns.values,oil.isnull().any().values))

lastdate = train.iloc[train.shape[0]-1].date
```

[PyFormat: Using % and .format\(\) for great good!](https://pyformat.info/)

[Python strftime reference](http://strftime.org/)

