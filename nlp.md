# NLP

* [https://blog.floydhub.com/ten-trends-in-deep-learning-nlp/](https://blog.floydhub.com/ten-trends-in-deep-learning-nlp/)
* [https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel](https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel)
* [https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/](https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/)
* [https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384](https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384)

## For the project

* Shall we just use TF-idf or TextRank \([如何用Python提取中文关键词？](https://zhuanlan.zhihu.com/p/31870596)\)
* Seems the no. of documents are not large enough to justify the replacement of manual method?
* Compare any new circular with all the old ones as a way of "new top detection"?
  * seems spaCy can do
  * can do using [Gensim](https://radimrehurek.com/gensim/tut3.html)
  * using [Doc2Vec](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb)
  * using [Document Embeddings](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md)?
* [**Train a new entity type "topic"**](https://spacy.io/usage/training#example-new-entity-type) **and use NER to find all topics**
  * NER SOTA is [around .9 F1 by flair](https://github.com/zalandoresearch/flair)
* MVP
  * just keyword search
  * td-idf for new topic
  * NER for new topic
* Keyword extraction 
  * using [gensim](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/summarization_tutorial.ipynb)
  * [Automated Keyword Extraction – TF-IDF, RAKE, and TextRank](http://www.tiernok.com/posts/automated-keyword-extraction-tf-idf-rake-and-textrank.html)

## Topic Modeling

* [LDA2vec: Word Embeddings in Topic Models](https://towardsdatascience.com/lda2vec-word-embeddings-in-topic-models-4ee3fc4b2843)
* [Combing LDA and Word Embeddings for topic modeling](https://towardsdatascience.com/combing-lda-and-word-embeddings-for-topic-modeling-fe4a1315a5b4)
* [https://github.com/bmabey/pyLDAvis](https://github.com/bmabey/pyLDAvis)
* [https://github.com/bigartm/bigartm](https://github.com/bigartm/bigartm) \(422 stars only\)
* [Topic Modelling in Python with NLTK and Gensim](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)
* [Complete Guide to Topic Modeling - NLP-FOR-HACKERS](https://nlpforhackers.io/topic-modeling/amp/)
* [gensim](https://github.com/RaRe-Technologies/gensim): 9000 stars
* [直觀理解 LDA \(Latent Dirichlet Allocation\) 與文件主題模型](https://medium.com/@tengyuanchang/%E7%9B%B4%E8%A7%80%E7%90%86%E8%A7%A3-lda-latent-dirichlet-allocation-%E8%88%87%E6%96%87%E4%BB%B6%E4%B8%BB%E9%A1%8C%E6%A8%A1%E5%9E%8B-ab4f26c27184)
* [Discovering and Classifying In-app Message Intent at Airbnb](https://medium.com/airbnb-engineering/discovering-and-classifying-in-app-message-intent-at-airbnb-6a55f5400a0c)
* [Topic Modeling with LSA, PLSA, LDA & lda2Vec](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)
* [Topic Modeling with Scikit Learn \(LDA and NMF\)](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730)
* [Using LDA Topic Models as a Classification Model Input](https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28)
* [Improving the Interpretation of Topic Models](https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847d)

### LDA

* [How we Changed Unsupervised LDA to Semi-Supervised GuidedLDA](https://www.freecodecamp.org/news/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164/)
* [Refining an LDA model or results \(Azure\)](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/latent-dirichlet-allocation?fireglass_rsn=true#refining-an-lda-model-or-results)
* [gensim LDA模型的优劣评估](https://zhuanlan.zhihu.com/p/33053850)
* \*\*\*\*[如何用Python从海量文本抽取主题？\(good intro\)](https://zhuanlan.zhihu.com/p/28992175)
* \*\*\*\*[**Topic Modeling with Gensim \(Python\)**](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)\*\*\*\*
*  [（四）Gensim简介、LDA编程实现、LDA主题提取效果图展示](https://zhuanlan.zhihu.com/p/28830480)
* [Topic modeling visualization – How to present the results of LDA models?](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
* [LDA in Python – How to grid search best topic models?](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/)
* [Introduction to Topic Modeling in Python](http://chdoig.github.io/pygotham-topic-modeling/#/)

#### No. of topics

* [Evaluation of Topic Modeling: Topic Coherence](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/) `u_mass` and `c_v`

> If you see the same keywords being repeated in multiple topics, it’s probably a sign that the ‘k’ is too large.

* [怎么确定LDA的topic个数？](https://www.zhihu.com/question/32286630)
* [Select number of topics for LDA model](https://cran.r-project.org/web/packages/ldatuning/vignettes/topics.html)
* Section "Finding out the optimal number of topics" in [this tutorial by gensim](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim_news_classification.ipynb)

## TD-IDF

* [Good intro](https://taweihuang.hpd.io/2017/03/01/tfidf/)

## PDFMiner

[如何用Python批量提取PDF文本内容？](https://zhuanlan.zhihu.com/p/34819237)

## Notes

According to [bert-as-service](https://bert-as-service.readthedocs.io/en/latest/section/what-is-it.html):

> **BERT** is a NLP model [developed by Google](https://github.com/google-research/bert) for pre-training language representations. It leverages an enormous amount of plain text data publicly available on the web and is trained in an unsupervised manner. Pre-training a BERT model is a fairly expensive yet one-time procedure for each language. Fortunately, Google released several pre-trained models where [you can download from here](https://github.com/google-research/bert#pre-trained-models).
>
> **Sentence Encoding/Embedding** is a upstream task required in many NLP applications, e.g. sentiment analysis, text classification. The goal is to represent a variable length sentence into a fixed length vector, e.g. hello world to \[0.1, 0.3, 0.9\]. Each element of the vector should “encode” some semantics of the original sentence.

According to [Google Research](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=xiYrZKaHwV81):

> In an existing pipeline, BERT can replace text embedding layers like ELMO and GloVE. Alternatively, [finetuning](http://wiki.fast.ai/index.php/Fine_tuning) BERT can provide both an accuracy boost and faster training time in many cases.

## Can be used by audit potentially

* text classification
  * can do multi-label using BERT \([Hugging Face](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)\), [fastText](https://fasttext.cc/docs/en/supervised-tutorial.html#multi-label-classification)
* sentiment analysis \(in fact is a form of text classification\)
* named entity recognition
* question answering
* summarisation
* relation extraction
* semantic parsing
* relation prediction
* sematic textual similarity
* sematic role labelling
* taxonomy learning

## Benchmark

* DecaNLP
* GLUE

## Full list of task

* [https://nlpprogress.com/](https://nlpprogress.com/)



## Analyzing Text Data

[Script for basic text analysis \(top nouns, verbs, entities and text similarity\) using spaCy](https://towardsdatascience.com/reliving-avengers-infinity-war-with-spacy-and-natural-language-processing-2abcb48e4ba1)

## Tutorial

[Building NLP Classifiers Cheaply With Transfer Learning and Weak Supervision](https://towardsdatascience.com/a-technique-for-building-nlp-classifiers-efficiently-with-transfer-learning-and-weak-supervision-a8e2f21ca9c8)

## Text Classification

* [Text Classification Guide by Google](https://developers.google.com/machine-learning/guides/text-classification/)
* [fast-bert](https://github.com/kaushaltrivedi/fast-bert)



## Library

* [https://fasttext.cc/docs/en/supervised-tutorial.html\#multi-label-classification](https://fasttext.cc/docs/en/supervised-tutorial.html#multi-label-classification)
* what [fast.ai library](https://docs.fast.ai/text.html) can do:
  * language model
  * RNN classifier
* spaCy is still actively developed
  * Similarity
  * Text Classification
* [When should I use what?](https://spacy.io/usage/facts-figures#comparison-usage)

## Good Resources

* [The Best and Most Current of Modern Natural Language Processing](https://medium.com/huggingface/the-best-and-most-current-of-modern-natural-language-processing-5055f409a1d1)



