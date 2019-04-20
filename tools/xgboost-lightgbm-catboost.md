# XGBoost, LightGBM, CatBoost

## LightGBM

> LightGBM offers good accuracy with integer-encoded categorical features. LightGBM applies [Fisher \(1958\)](http://www.csiss.org/SPACE/workshops/2004/SAC/files/fisher.pdf) to find the optimal split over categories as [described here](https://lightgbm.readthedocs.io/en/latest/Features.html#optimal-split-for-categorical-features). This often performs better than one-hot encoding.

## CatBoost

* Provides [CatBoost Viewer](https://catboost.ai/docs/features/visualization_catboost-viewer.html) to analyze training process
* Categorical features are used to build new numeric features based on categorical features and their combinations. See the [Transforming categorical features to numerical features](https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html#algorithm-main-stages_cat-to-numberic) section for more details.

