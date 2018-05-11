# fastai

When training using the fast.ai library, it prints out 3 numbers each cycle:

* training loss
* validation loss
* accuracy

If the validation loss is much lower than training loss, the model is under fitting. It means the cycle length is too short - the log pop up before reaching to the bottom.

'precompute = true' is used as a shortcut if the dataset is large as it is much faster - about 10x faster.

If a smaller batch size is used, the gradient is calculated using less number of images so it is less accurate as it is more volatile. You can try to re-run the learning rate finder to see if the best learning rate changed but it shouldn't make a huge difference as the learning rate differ exponentially.



