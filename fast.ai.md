# Fast.ai

## Deep Learning v2

### Lesson 2

When training using the fast.ai library, it prints out 3 numbers each cycle:

* training loss
* validation loss
* accuracy

If the validation loss is much lower than training loss, the model is under fitting. It means the cycle length is too short - the log pop up before reaching to the bottom.

'precompute = true' is used as a shortcut if the dataset is large as it is much faster - about 10x faster.

[https://sanctum.geek.nz/arabesque/zooming-tmux-panes](https://sanctum.geek.nz/arabesque/zooming-tmux-panes)/

### Lesson 3

[https://chrome.google.com/webstore/detail/curlwget/jmocjfidanebdlinpbcdkcmgdifblncg](https://chrome.google.com/webstore/detail/curlwget/jmocjfidanebdlinpbcdkcmgdifblncg)

to show the symbolic link: `ls -l`

data augmentation doesn't work if `precompute = True`. I don't understand why though.

Softmax likes picking one class. So it is ridiculous to use it for multi-label classification

Great book: [https://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793](https://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793)

### Lesson 4

Neural nets really like standardized variables. This can done by setting `do_scale`=`True` when calling `prod_df`.

