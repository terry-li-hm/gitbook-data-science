# Computer Vision

* [Going deep into image classification – Towards Data Science](https://towardsdatascience.com/an-overview-of-image-classification-networks-3fb4ff6fa61b)
* [GitHub新項目：輕鬆使用多種預訓練卷積網絡抽取圖像特徵 \| 機器之心](https://www.jiqizhixin.com/articles/2018-04-16-3)
* [An Overview of ResNet and its Variants – Towards Data Science](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
* [覽輕量化卷積神經網絡：SqueezeNet、MobileNet、ShuffleNet、Xception \| 機器之心](https://www.jiqizhixin.com/articles/2018-01-08-6)
* [Google Landmark Retrieval Challenge \| Kaggle](https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/57855)

### Object Detection

[Detectron精讀系列之一：學習率的調節和踩坑 \| 機器之心](https://www.jiqizhixin.com/articles/Detectron)

[matterport/Mask\_RCNN: Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow](https://github.com/matterport/Mask_RCNN)



* [Object detection: an overview in the age of Deep Learning](https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning/)
* [Is Google Tensorflow Object Detection API the easiest way to implement image recognition?](https://medium.com/towards-data-science/is-google-tensorflow-object-detection-api-the-easiest-way-to-implement-image-recognition-a8bd1f500ea0)
* [How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)
* [深度学习目标检测模型全面综述：Faster R-CNN、R-FCN和SSD](https://www.jiqizhixin.com/articles/2017-09-18-7)
* [Deep Learning for Object Detection: A Comprehensive Review](https://medium.com/towards-data-science/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)

A tricked found by Jeremy to avoid overfitting is to train a network with small images for few epochs and then train it using larger images. It is only applicable to architectures that can take arbitrary image sizes and thus not applicable to VGG.

If a smaller batch size is used, the gradient is calculated using less number of images so it is less accurate as it is more volatile. You can try to re-run the learning rate finder to see if the best learning rate changed but it shouldn't make a huge difference as the learning rate differ exponentially. \(per Jeremy\)

