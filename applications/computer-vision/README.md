# Computer Vision

* [Going deep into image classification – Towards Data Science](https://towardsdatascience.com/an-overview-of-image-classification-networks-3fb4ff6fa61b)
* [GitHub新項目：輕鬆使用多種預訓練卷積網絡抽取圖像特徵 \| 機器之心](https://www.jiqizhixin.com/articles/2018-04-16-3)
* [An Overview of ResNet and its Variants – Towards Data Science](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
* [覽輕量化卷積神經網絡：SqueezeNet、MobileNet、ShuffleNet、Xception \| 機器之心](https://www.jiqizhixin.com/articles/2018-01-08-6)
* [Google Landmark Retrieval Challenge \| Kaggle](https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/57855)
* [ML Practicum: Image Classification  \|  Machine Learning Practica  \|  Google Developers](https://developers.google.com/machine-learning/practica/image-classification/)

### Object Detection

* [Detectron精讀系列之一：學習率的調節和踩坑 \| 機器之心](https://www.jiqizhixin.com/articles/Detectron)
* [Detecting Pikachu in videos using Tensorflow Object Detection](https://towardsdatascience.com/detecting-pikachu-in-videos-using-tensorflow-object-detection-cd872ac42c1d)

[matterport/Mask\_RCNN: Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow](https://github.com/matterport/Mask_RCNN)



* [Object detection: an overview in the age of Deep Learning](https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning/)
* [Is Google Tensorflow Object Detection API the easiest way to implement image recognition?](https://medium.com/towards-data-science/is-google-tensorflow-object-detection-api-the-easiest-way-to-implement-image-recognition-a8bd1f500ea0)
* [How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)
* [深度学习目标检测模型全面综述：Faster R-CNN、R-FCN和SSD](https://www.jiqizhixin.com/articles/2017-09-18-7)
* [Deep Learning for Object Detection: A Comprehensive Review](https://medium.com/towards-data-science/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)
* [Going beyond the bounding box with semantic segmentation](https://thegradient.pub/semantic-segmentation/)

A tricked found by Jeremy to avoid overfitting is to train a network with small images for few epochs and then train it using larger images. It is only applicable to architectures that can take arbitrary image sizes and thus not applicable to VGG.

If a smaller batch size is used, the gradient is calculated using less number of images so it is less accurate as it is more volatile. You can try to re-run the learning rate finder to see if the best learning rate changed but it shouldn't make a huge difference as the learning rate differ exponentially. \(per Jeremy\)

* [Taskonomy \| Stanford](http://taskonomy.stanford.edu/)
* [An overview of semantic image segmentation.](https://www.jeremyjordan.me/semantic-segmentation/)
* [DIY Deep Learning Projects – Towards Data Science](https://towardsdatascience.com/diy-deep-learning-projects-c2e0fac3274f)
* [Spotting Image Manipulation with AI \| Adobe Blog](https://theblog.adobe.com/spotting-image-manipulation-ai/)
* [How to build a custom face recognition dataset - PyImageSearch](https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/)
* [如何妙笔勾檀妆：像素级语义理解](https://zhuanlan.zhihu.com/p/34607294)
* [CVPR 2018 Best Paper Taskonomy 作者解读](https://zhuanlan.zhihu.com/p/38425434)
* [Measuring size of objects in an image with OpenCV - PyImageSearch](https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/)
* [攜程李翔：深度學習在酒店圖像智能化上的一系列應用 \| 雷鋒網](https://www.leiphone.com/news/201806/wQMFVsUdkuo9zF1m.html)
* [facebookresearch/video-nonlocal-net: Non-local Neural Networks for Video Classification](https://github.com/facebookresearch/video-nonlocal-net)
* [vipstone/faceai: 一款入門級的人臉、視頻、文字檢測以及識別的項目.](https://github.com/vipstone/faceai)
* [圖像分類比賽中，你可以用如下方案舉一反三 \| 雷鋒網](https://www.leiphone.com/news/201806/tOdey3hbTo0dQIJ5.html)
* [Unsupervised Deep Learning Algorithms for Computer Vision](https://www.analyticsvidhya.com/blog/2018/06/unsupervised-deep-learning-computer-vision/)
* [How I built a Self Flying Drone to track People in under 50 lines of code](https://medium.com/nanonets/how-i-built-a-self-flying-drone-to-track-people-in-under-50-lines-of-code-7485de7f828e)
* [What Image Classifiers Can Do About Unknown Objects « Pete Warden's blog](https://petewarden.com/2018/07/06/what-image-classifiers-can-do-about-unknown-objects/)

