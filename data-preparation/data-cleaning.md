# Data Cleaning

## Abnormal values

For the Bulldozers Kaggle competition, Jeremy noted from the plot below that there are many samples of which `YearMade` is 1000:

![](../.gitbook/assets/image%20%2814%29.png)

He guesses these are in fact missing values and [change them into ](https://youtu.be/0v93qHDqq_g?t=1h31m25s)`1950`:

```python
df_raw.YearMade[df_raw.YearMade<1950] = 1950
```

Why he does so? Aren't 1000 and 1950 are the same for tree-based algorithm \(as the split point should be the same if the rest are &gt; 1950?

## IDs

[Jeremy doesn't drop the IDs](https://youtu.be/0v93qHDqq_g?t=5832). He just treat them as normal categorical variables. I think the reason is that the IDs may also provide information \(e.g. splitting at some point of the IDs provides information gain\).



![](../.gitbook/assets/image%20%281%29.png)

![](../.gitbook/assets/image%20%2847%29.png)

![](../.gitbook/assets/image%20%2862%29.png)

