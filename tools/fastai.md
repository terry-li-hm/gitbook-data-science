# fastai

When training using the fast.ai library, it prints out 3 numbers each cycle:

* training loss
* validation loss
* accuracy

If the validation loss is much lower than training loss, the model is under fitting. It means the cycle length is too short - the log pop up before reaching to the bottom.

`precompute = True` is used as a shortcut if the dataset is large as it is much faster - about 10x faster.

data augmentation doesn't work if `precompute = True`. I don't understand why though.

Neural nets really like standardized variables. This can done by setting `do_scale`=`True` when calling `prod_df`.  


* [FastAI Library \| Myndbook](https://myndbook.com/view/6278)
* [ExpressForce/Nik\_genetflix.ipynb at master · SVAI/ExpressForce](https://github.com/SVAI/ExpressForce/blob/master/Nik_genetflix.ipynb)
* [Training metrics as notifications on mobile using Callbacks - Part 2 & Alumni - Deep Learning Course Forums](http://forums.fast.ai/t/training-metrics-as-notifications-on-mobile-using-callbacks/17330)

