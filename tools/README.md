# Tools

* [Zooming tmux panes \| Arabesque](https://sanctum.geek.nz/arabesque/zooming-tmux-panes/)
* [CurlWget - Chrome Web Store](https://chrome.google.com/webstore/detail/curlwget/jmocjfidanebdlinpbcdkcmgdifblncg)



## Theano

```python
# For the following error:
# To use MKL 2018 with Theano you MUST set "MKL_THREADING_LAYER=GNU"

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
```

