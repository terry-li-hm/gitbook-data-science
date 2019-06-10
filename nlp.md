# NLP

* [https://blog.floydhub.com/ten-trends-in-deep-learning-nlp/](https://blog.floydhub.com/ten-trends-in-deep-learning-nlp/)
* [https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel](https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel)
* [https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/](https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/)
* [https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384](https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384)

## For the project

* Seems the no. of documents are not large enough to justify the replacement of manual method?
* Possible solutions
  * Topic modeling
  * Identify topics and count occurrence
  * * Keyword search \(cannot find new topics automatically\)
    * TF-IDF
    * RAKE
    * TextRank
    * Train a NER model to identify topics
    * Text classification \(cannot find new topics automatically\) + document similarity to find new topics\)
  * Detecting new topics by comparing the similarity of new circulars to all old circulars



* Compare any new circular with all the old ones as a way of "new top detection"?
  * seems spaCy can do
  * can do using [Gensim](https://radimrehurek.com/gensim/tut3.html)
  * using [Doc2Vec](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb)
  * using [Document Embeddings](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md)?
  * using [Gensim](https://nlpforhackers.io/topic-modeling/) \(`similarities` module\)
  * [using R package text2vec](http://text2vec.org/similarity.html)
* [**Train a new entity type "topic"**](https://spacy.io/usage/training#example-new-entity-type) **and use NER to find all topics**
  * NER SOTA is [around .9 F1 by flair](https://github.com/zalandoresearch/flair)
* Keyword extraction 
  * using [gensim](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/summarization_tutorial.ipynb)
  * [Automated Keyword Extraction – TF-IDF, RAKE, and TextRank](http://www.tiernok.com/posts/automated-keyword-extraction-tf-idf-rake-and-textrank.html)
  * [textacy.keyterms](https://chartbeat-labs.github.io/textacy/getting_started/quickstart.html#)
  * [TextRank](https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0)
  * \*\*\*\*[rake-nltk](https://github.com/csurfer/rake-nltk)
  * [How to extract keywords from text with TF-IDF and Python’s Scikit-Learn](https://www.freecodecamp.org/news/how-to-extract-keywords-from-text-with-tf-idf-and-pythons-scikit-learn-b2a0f3d7e667/)
  * [PyTextRank](https://medium.com/@aneesha/beyond-bag-of-words-using-pytextrank-to-find-phrases-and-summarize-text-f736fa3773c5)
  * [如何用Python提取中文关键词？](https://zhuanlan.zhihu.com/p/31870596)
* [Anomaly detection in Tweets: Clustering & Proximity based approach](https://medium.com/swlh/anomaly-detection-in-tweets-clustering-proximity-based-approach-58f8c22eed1e) \(just 7 claps\)

## Topic Modeling

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
* [Example codes for implementing LDA and NMF using sk-learn](https://towardsdatascience.com/topic-modeling-for-everybody-with-google-colab-2f5cdc99a647)
* [Complete Guide to Topic Modeling](https://nlpforhackers.io/topic-modeling/)

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
* [Why Latent Dirichlet Allocation Sucks](https://eigenfoo.xyz/lda-sucks/)
* [Building an Article Recommender using LDA](https://towardsdatascience.com/lets-build-an-article-recommender-using-lda-f22d71b7143e)
* [https://github.com/bmabey/pyLDAvis](https://github.com/bmabey/pyLDAvis) \(The size of the circle is determined by the prevalence of the topic.\)

#### No. of topics

* [Evaluation of Topic Modeling: Topic Coherence](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/) `u_mass` and `c_v`

> If you see the same keywords being repeated in multiple topics, it’s probably a sign that the ‘k’ is too large.

* [怎么确定LDA的topic个数？](https://www.zhihu.com/question/32286630)
* [Select number of topics for LDA model](https://cran.r-project.org/web/packages/ldatuning/vignettes/topics.html)
* Section "Finding out the optimal number of topics" in [this tutorial by gensim](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim_news_classification.ipynb)

According to Airbnb:

> We determined the number of topics \(hyperparameter _K_\) in LDA to be the one generating the highest [coherence score](http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) on the validation set.

Why need to carve out validation set for LDA?

#### Application by [Airbnb](https://medium.com/airbnb-engineering/discovering-and-classifying-in-app-message-intent-at-airbnb-6a55f5400a0c)

> To address this challenge, we set up our solutions in two phases: In **Phase 1**, we used a classic unsupervised approach — [Latent Dirichlet Allocation \(LDA\)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) — to discover potential topics \(intents\) in the large message corpus. In **Phase 2**, we moved to supervised learning techniques, but used the topics derived from Phase 1 as intent labels for each message. Specifically, we built a multi-class classification model using a canonical Convolutional Neural Network \(CNN\) architecture. The two phases create a powerful framework for us to accurately understand the text data on our messaging platform.

### lda2vec

By the [author](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/):

> ### Should I use lda2vec? <a id="should-i-use-lda2vec"></a>
>
> **Probably not!** At a practical level, if you want human-readable topics just use LDA \(checkout libraries in [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) and [gensim](https://radimrehurek.com/gensim/models/ldamodel.html)\). If you want machine-useable word-level features, use word2vec. But if you want to rework your own topic models that, say, jointly correlate an article’s topics with votes or predict topics over users then you might be interested in [lda2vec](https://github.com/cemoody/lda2vec).
>
> There are also a number of reasons not to use lda2vec: while the code has decent unit testing and reasonable documentation, it’s built to drive experimentation. It requires a huge amount of computation, and so I wouldn’t really try it without GPUs. Furthermore, I haven’t measured lda2vec’s performance against LDA and word2vec baselines – it might be worse or it might be better, and your mileage may vary.

* [LDA2vec: Word Embeddings in Topic Models](https://towardsdatascience.com/lda2vec-word-embeddings-in-topic-models-4ee3fc4b2843)

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

## BERT

As little as 1000 training samples are needed to train a good text classifier \(according to a [Github repo](https://github.com/Socialbird-AILab/BERT-Classification-Tutorial)\)

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



