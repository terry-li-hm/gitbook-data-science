# Computer Vision

* [Going deep into image classification – Towards Data Science](https://towardsdatascience.com/an-overview-of-image-classification-networks-3fb4ff6fa61b)
* [GitHub新項目：輕鬆使用多種預訓練卷積網絡抽取圖像特徵 \| 機器之心](https://www.jiqizhixin.com/articles/2018-04-16-3)
* [An Overview of ResNet and its Variants – Towards Data Science](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
* [覽輕量化卷積神經網絡：SqueezeNet、MobileNet、ShuffleNet、Xception \| 機器之心](https://www.jiqizhixin.com/articles/2018-01-08-6)
* [Google Landmark Retrieval Challenge \| Kaggle](https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/57855)
* [ML Practicum: Image Classification  \|  Machine Learning Practica  \|  Google Developers](https://developers.google.com/machine-learning/practica/image-classification/)
* [Reinventing the Wheel of Semantic Segmentation: – 100 Shades of Machine Learning – Medium](https://medium.com/100-shades-of-machine-learning/https-medium-com-100-shades-of-machine-learning-rediscovering-semantic-segmentation-part1-83e1462e0805)

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
* [CMU-Perceptual-Computing-Lab/openpose: OpenPose: Real-time multi-person keypoint detection library for body, face, and hands estimation](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [Drones taught to spot violent behavior in crowds using AI - The Verge](https://www.theverge.com/2018/6/6/17433482/ai-automated-surveillance-drones-spot-violent-behavior-crowds)
* [This Japanese AI security camera shows the future of surveillance will be automated - The Verge](https://www.theverge.com/2018/6/26/17479068/ai-guardman-security-camera-shoplifter-japan-automated-surveillance)
* [What are radiological deep learning models actually learning?](https://medium.com/@jrzech/what-are-radiological-deep-learning-models-actually-learning-f97a546c5b98)
* [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
* [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://eng.uber.com/coordconv/)
* [Attention on Pretrained-VGG16 for Bone Age \| Kaggle](https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age/notebook)
* [Meet RoboSat 🤖 🛰 – Points of interest](https://blog.mapbox.com/meet-robosat-af42530f163f)
* [One-shot object detection](http://machinethink.net/blog/object-detection/)
* [NVIDIA's Noise2Noise Technique Helps you Fix Bad Images in Milliseconds](https://www.analyticsvidhya.com/blog/2018/07/nvidias-noise2noise-technique-helps-you-fix-bad-images-in-milliseconds/)
* ['AI Guardman' - A Machine Learning Application that uses Pose Estimation to Detect Shoplifters](https://www.analyticsvidhya.com/blog/2018/06/ai-guardman-machine-learning-application-estimates-poses-detect-shoplifters/)
* [Training a Machine to Watch Soccer \| Caltech](http://www.caltech.edu/news/training-machine-watch-soccer-79455)
* [Multi-label classification with Keras - PyImageSearch](https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/)
* [ageitgey/face\_recognition: The world's simplest facial recognition api for Python and the command line](https://github.com/ageitgey/face_recognition)
* [How to handle mistakes while using AI to block attacks](https://elie.net/blog/ai/how-to-handle-mistakes-while-using-ai-to-block-attacks)
* [Build your own Computer Vision Model with the Latest TensorFlow Object Detection API Update](https://www.analyticsvidhya.com/blog/2018/07/build-computer-vision-model-tensorflow-object-detection-api/)
* [A new hope: AI for news media \| TechCrunch](https://techcrunch.com/2018/07/12/a-new-hope-ai-for-news-media/)
* [With AI, Your Apple Watch Could Flag Signs of Diabetes \| WIRED](https://www.wired.com/story/with-ai-your-apple-watch-could-flag-signs-of-diabetes/)
* [Cryptocurrencies leveraging Natural Language Processing for profit](https://blog.usejournal.com/cryptocurrencies-leveraging-natural-language-processing-for-profit-a072cc97d7e1)
* [Automatic tagging of clothing in E-Commerce, Using Tensorflow and GCP.](https://blog.usejournal.com/automatic-tagging-of-clothing-in-e-commerce-using-tensorflow-and-gcp-d2b623cd2a78)
* [Feature-wise transformations](https://distill.pub/2018/feature-wise-transformations/)
* [Building a Mask R-CNN Model for Detecting Car Damage \(Python codes\)](https://www.analyticsvidhya.com/blog/2018/07/building-mask-r-cnn-model-detecting-damage-cars-python/)
* [Swimming pool detection and classification using deep learning](https://medium.com/geoai/swimming-pool-detection-and-classification-using-deep-learning-aaf4a3a5e652)
* [image\_captioning\_with\_attention.ipynb - Colaboratory](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/image_captioning_with_attention.ipynb?linkId=54343050)
* [mrecos/signboardr: Extract text from sign board and tag as metadata](https://github.com/mrecos/signboardr)
* [Cutting-Edge Face Recognition is Complicated. These Spreadsheets Make it Easier.](https://towardsdatascience.com/cutting-edge-face-recognition-is-complicated-these-spreadsheets-make-it-easier-e7864dbf0e1a)
* [mrecos/signboardr: Extract text from sign board and tag as metadata](https://github.com/mrecos/signboardr)

