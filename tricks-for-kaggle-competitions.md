# Tricks for Kaggle Competitions

[Per Jeremy](https://youtu.be/3jl2h9hSRvc?t=52m19s):

> ...create a random forest where my dependent variable is “is it in the validation set” \(`is_valid`\). I’ve gone back and I’ve got my whole data frame with the training and validation all together and I’ve created a new column called `is_valid` which I’ve set to one and then for all of the stuff in the training set, I set it to zero. So I’ve got a new column which is just is this in the validation set or not and then I’m going to use that as my dependent variable and build a random forest. This is a random forest not to predict price but predict is this in the validation set or not. So if your variable were not time dependent, then it shouldn’t be possible to figure out if something is in the validation set or not.
>
> ```text
> df_ext = df_keep.copy()
> df_ext['is_valid'] = 1
> df_ext.is_valid[:n_trn] = 0
> x, y, nas = proc_df(df_ext, 'is_valid')
> ```
>
> This is a great trick in Kaggle because they often won’t tell you whether the test set is a random sample or not. So you could put the test set and training set together, create a new column called `is_test` and see if you can predict it. If you can, you don’t have a random sample which means you have to figure out how to create a validation set from it.

