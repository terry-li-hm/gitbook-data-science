# Fast.ai

## Deep Learning v2

### Lesson 2

A tricked found by Jeremy to avoid overfitting is to train a network with small images for few epochs and then train it using larger images. It is only applicable to architectures that can take arbitrary image sizes and thus not applicable to VGG.

When training using the fast.ai library, it prints out 3 numbers each cycle:

* training loss
* validation loss
* accuracy

If the validation loss is much lower than training loss, the model is under fitting. It means the cycle length is too short - the log pop up before reaching to the bottom.

When you are happy with your model, treat also the validation set as part of the training set as re-do the training process to use 100% of the training set to further improve the result.

'precompute = true' is used as a shortcut if the dataset is large as it is much faster - about 10x faster.

If a smaller batch size is used, the gradient is calculated using less number of images so it is less accurate as it is more volatile. You can try to re-run the learning rate finder to see if the best learning rate changed but it shouldn't make a huge difference as the learning rate differ exponentially.

[https://sanctum.geek.nz/arabesque/zooming-tmux-panes](https://sanctum.geek.nz/arabesque/zooming-tmux-panes)/

### Lesson 3

[https://chrome.google.com/webstore/detail/curlwget/jmocjfidanebdlinpbcdkcmgdifblncg](https://chrome.google.com/webstore/detail/curlwget/jmocjfidanebdlinpbcdkcmgdifblncg)

to show the symbolic link: `ls -l`

data augmentation doesn't work if `precompute = True`. I don't understand why though.

Softmax likes picking one class. So it is ridiculous to use it for multi-label classification

Great book: [https://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793](https://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793)

### Lesson 4

Neural nets really like standardized variables. This can done by setting `do_scale`=`True` when calling `prod_df`.

