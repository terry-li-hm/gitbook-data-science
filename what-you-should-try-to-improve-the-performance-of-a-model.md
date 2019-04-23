# What you should try to improve the performance of a model

* Create many features \(what the 1st place of the Rossmann competition did\). However, the 3rd place did almost no feature engineering. Instead, he used deep learning. It seems \(from what [he said](https://youtu.be/YSFG_W8JxBo?t=42m52s)\) that Jeremy thinks both are possible ways.

## Remove redundant features

Why? We’ve already seen how variables which are basically measuring the same thing can confuse our variable importance. They can also make our random forest slightly less good because it requires more computation to do the same thing and there’re more columns to check. So we are going to do some more work to try and remove redundant features. How? Refer to [this notebook by Jeremy](https://render.githubusercontent.com/view/ipynb?commit=b5d273c84c0894a10ed290a64d64fb92b2d43c4f&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6661737461692f6661737461692f623564323733633834633038393461313065643239306136346436346662393262326434336334662f636f75727365732f6d6c312f6c6573736f6e322d72665f696e746572707265746174696f6e2e6970796e62&nwo=fastai%2Ffastai&path=courses%2Fml1%2Flesson2-rf_interpretation.ipynb&repository_id=102973646&repository_type=Repository#Removing-redundant-features).

## If the data is temporal, use only the recent data for training?

[Jeremy said](https://youtu.be/3jl2h9hSRvc?t=1146) this is something that you may try \(i.e. may or may not work\). One drawback is that it used less data to train. One alternative for tree-based algorithm is to give more weight to the recent rows, i.e. they get higher chance of being selected during bootstrapping.



