# Data Cleansing

[The Mechanics of Machine Learning](https://mlbook.explained.ai/prep.html):

> Don't look at the data first and then decide on a definition of anomalous. You risk removing or altering data simply because it looks inconvenient or looks like it might confuse the model.

> How much we care about cleaning up the data depends on the model we're using and whether the offending values are in predictor variables \(features\) or the target. One of the advantages of RFs is that they deal gracefully with errors and outliers in the predictor variables. RFs behave like nearest-neighbor models and feature outliers are partitioned off into lonely corners of the feature space automatically. Anomalous values in the target variable are also not a problem, unless they lead to inconsistencies, samples with the same or similar feature vectors but huge variation in the target values. No model deals well with inconsistent training data.

