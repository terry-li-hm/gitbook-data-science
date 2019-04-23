# Validation Set

Use `n_valid = len(df_test)` to ensure that the size of the validation set is always same as the test set. This is what [Jeremy does](https://youtu.be/YSFG_W8JxBo?t=1685).

## How to construct a good validation set

[Per Jeremy](https://youtu.be/3jl2h9hSRvc?t=24m1s):

1. Build few different models \(e.g. different hyper parameters, different training set\).
2. Plot a scatter plot of test set score and validation set score. It should gives a straight line, i.e. the validation set score is a good proxy of the test set score. Note that when getting the test set score, you train the modeling by feeding in both train and validation set.
3. If it is not a straight line, construct the validation set using a different strategy. Never change your test set. The idea is that you are calibrating the validation test.

### Size of Validation Set

[Jeremy had a discussion on this ](https://youtu.be/O5F9vR2CNYI?t=5m42s)in this class. But not quite understand.

