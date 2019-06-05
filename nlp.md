# NLP

* [https://blog.floydhub.com/ten-trends-in-deep-learning-nlp/](https://blog.floydhub.com/ten-trends-in-deep-learning-nlp/)
* [https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel](https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel)
* [https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/](https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/)
* [https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384](https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384)

## Topic Modeling

* [LDA2vec: Word Embeddings in Topic Models](https://towardsdatascience.com/lda2vec-word-embeddings-in-topic-models-4ee3fc4b2843)
* [Combing LDA and Word Embeddings for topic modeling](https://towardsdatascience.com/combing-lda-and-word-embeddings-for-topic-modeling-fe4a1315a5b4)
* [https://github.com/bmabey/pyLDAvis](https://github.com/bmabey/pyLDAvis)
* [https://github.com/bigartm/bigartm](https://github.com/bigartm/bigartm) \(422 stars only\)
* [Topic Modelling in Python with NLTK and Gensim](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)
* [Complete Guide to Topic Modeling - NLP-FOR-HACKERS](https://nlpforhackers.io/topic-modeling/amp/)
* [gensim](https://github.com/RaRe-Technologies/gensim): 9000 stars



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

[Text Classification Guide by Google](https://developers.google.com/machine-learning/guides/text-classification/)



## Library

* [https://fasttext.cc/docs/en/supervised-tutorial.html\#multi-label-classification](https://fasttext.cc/docs/en/supervised-tutorial.html#multi-label-classification)
* what [fast.ai library](https://docs.fast.ai/text.html) can do:
  * language model
  * RNN classifier

## Good Resources

* [The Best and Most Current of Modern Natural Language Processing](https://medium.com/huggingface/the-best-and-most-current-of-modern-natural-language-processing-5055f409a1d1)



