# Key

## The first thing to do for any supervised learning problem

Just follow through the [lesson 1 ](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson1-rf.ipynb)of the Jeremy's course. Per [Jeremy](https://youtu.be/CzdWqFTmn0Y?t=4546) it is a standardized approach that works as a baseline.

## Use a sample of training set \(if the training set is large\)

Jeremy [strongly recommends ](https://youtu.be/blyXCk4sgEg?t=4858)you a sample of the training set to experiment as it would be much faster. Just use a sample size that is large enough to give you good enough accuracy.



[Jeremy said](https://youtu.be/YSFG_W8JxBo?t=1h7m20s):

> I always look at feature importance first in practice. Whether I’m working on a Kaggle competition or a real world project, I build a random forest as fast as I can, trying to get it to the point that is significantly better than random but doesn’t have to be much better than that. And the next thing I do is to plot the feature importance.
>
>  The feature importance tells us in this random forest, which columns mattered. We have dozens of columns in this dataset, and here, we are picking out the top 10. `rf_feat_importance` is part of Fast.ai library which takes a model `m` and dataframe `df_trn` \(because we need to know names of columns\) and it will give you back a Pandas dataframe showing you in order of importance how important each column was.

> ```text
> fi = rf_feat_importance(m, df_trn); fi[:10]
> ```

![](.gitbook/assets/image%20%2854%29.png)

> ```text
> fi.plot('cols', 'imp', figsize=(10,6), legend=False);
> ```

![](.gitbook/assets/image%20%2851%29.png)

Since `fi` is a `DataFrame`, we can use `DataFrame` plotting commands \[[1:09:00](https://youtu.be/YSFG_W8JxBo?t=1h9m)\]. The important thing is to see that some columns are really important and most columns do not really matter at all. In nearly every dataset you use in real life, this is what your feature importance is going to look like. There is only a handful of columns that you care about, and this is why Jeremy always starts here. At this point, in terms of looking into learning about this domain of heavy industrial equipment auctions, we only have to care about learning about the columns which matter. Are we going to bother learning about `Enclosure`? Depends whether `Enclosure` is important. It turns out that it appears in top 10, so we are going to have to learn about `Enclosure`.

We can also plot this as a bar plot:

```text
def plot_fi(fi): 
  return fi.plot('cols','imp','barh', figsize=(12,7), legend=False)
```

```text
plot_fi(fi[:30]);
```

![](https://cdn-images-1.medium.com/max/1200/1*ZegWuJLPnlJnYOOSR9wgvw.png)

The most important thing to do with this is to now sit down with your client, your data dictionary, or whatever your source of information is and say to then “okay, tell me about `YearMade`. What does that mean? Where does it come from?” \[[1:10:31](https://youtu.be/YSFG_W8JxBo?t=1h10m31s)\] Plot lots of things like histogram of `YearMade` and scatter plot of `YearMade` against price and learn everything you can because `YearMade` and `Coupler_System` — they are the things that matter.

What will often happen in real-world projects is that you sit with the the client and you’ll say “it turns out the `Coupler_System` is the second most important thing” and they might say “that makes no sense.” That doesn’t mean that there is a problem with your model, it means there is a problem with their understanding of the data they gave you.





[Jeremy suggests ](https://youtu.be/0v93qHDqq_g?t=29m30s)to focus the feature engineering effort on the top features of the feature importance chart.

## Should we do one-hot encoding?

[Per Jeremy](https://youtu.be/0v93qHDqq_g?t=49m15s), it is kind of necessary for many machine learning algorithm. But for tree-based algorithms, it is not necessary. It may or may not improve the performance. It is something that you can try. However, it still worth doing so as the feature importance with one-hot encoding may tell you that a particular level of a categorical variable is very informative. However, you won't for one-hot encoding for high cardinality as it will leads to computation issue. The magic number is cardinality of 6 to 7.

Jeremy doesn't keep the original variable after one-hot encoding it.

If you have actually made an effort to turn your ordinal variables into proper ordinals, using `proc_df` can destroy that. The simple way to avoid that is if we know that we always want to use the codes for usage band, you could just go ahead and replace it:

![](.gitbook/assets/image%20%285%29.png)



