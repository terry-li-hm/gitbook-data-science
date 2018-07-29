# NLP

## Word Embedding

A word embedding is an approach to provide a dense vector representation of words that capture something about their meaning.

Word embeddings are an improvement over simpler bag-of-word model word encoding schemes like word counts and frequencies that result in large and sparse vectors \(mostly 0 values\) that describe documents but not the meaning of the words.

Word embeddings work by using an algorithm to train a set of fixed-length dense and continuous-valued vectors based on a large corpus of text. Each word is represented by a point in the embedding space and these points are learned and moved around based on the words that surround the target word.

It is defining a word by the company that it keeps that allows the word embedding to learn something about the meaning of words. The vector space representation of the words provides a projection where words with similar meanings are locally clustered within the space.

The use of word embeddings over other text representations is one of the key methods that has led to breakthrough performance with deep neural networks on problems like machine translation.

Word2vec is one algorithm for learning a word embedding from a text corpus.

There are two main training algorithms that can be used to learn the embedding from text; they are continuous bag of words \(CBOW\) and skip grams.

We will not get into the algorithms other than to say that they generally look at a window of words for each target word to provide context and in turn meaning for words. The approach was developed by Tomas Mikolov, formerly at Google and currently at Facebook.



* [Google AI Blog: Text summarization with TensorFlow](https://ai.googleblog.com/2016/08/text-summarization-with-tensorflow.html?m=1)
* [**fast.ai NLP · Practical NLP**](http://nlp.fast.ai/)
* [Visualizing A Neural Machine Translation Model \(Mechanics of Seq2seq Models With Attention\) – Jay Alammar – Visualizing machine learning one concept at a time](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
* [Intuitive Understanding of Word Embeddings: Count Vectors to Word2Vec](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
* [How to get started in NLP – Towards Data Science](https://towardsdatascience.com/how-to-get-started-in-nlp-6a62aa4eaeff)
* [How to solve 90% of NLP problems: a step-by-step guide](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e)
* [Data Analysis & XGBoost Starter \(0.35460 LB\) \| Kaggle](https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb)
* [Bag of Words Meets Bags of Popcorn \| Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial#description)
* [Working With Text Data — scikit-learn 0.19.1 documentation](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
* [sloria/TextBlob: Simple, Pythonic, text processing--Sentiment analysis, part-of-speech tagging, noun phrase extraction, translation, and more.](https://github.com/sloria/textblob)
* [Getting Started with spaCy for Natural Language Processing](https://www.kdnuggets.com/2018/05/getting-started-spacy-natural-language-processing.html)
* [How I lost a silver medal in Kaggle’s Mercari Price Suggestion Challenge using CNNs and Tensorflow](https://medium.com/unstructured/how-i-lost-a-silver-medal-in-kaggles-mercari-price-suggestion-challenge-using-cnns-and-tensorflow-4013660fcded)
* [Understanding Feature Engineering \(Part 4\) — Deep Learning Methods for Text Data](https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)
* [fastText/pretrained-vectors.md at master · facebookresearch/fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
* [Kyubyong/nlp\_tasks: Natural Language Processing Tasks and References](https://github.com/Kyubyong/nlp_tasks)
* [xiamx/awesome-sentiment-analysis: 😀😄😂😭 A curated list of Sentiment Analysis methods, implementations and misc. 😥😟😱😤](https://github.com/xiamx/awesome-sentiment-analysis)
* [The Essential NLP Guide for data scientists \(codes for top 10 NLP tasks\)](https://www.analyticsvidhya.com/blog/2017/10/essential-nlp-guide-data-scientists-top-10-nlp-tasks/)
* [How to solve 90% of NLP problems: a step-by-step guide](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e)
* [What is TF-IDF? The 10 minute guide](http://michaelerasm.us/post/tf-idf-in-10-minutes/)
* [NLP: Any libraries/dictionaries out there for fixing common spelling errors? - Part 2 & Alumni - Deep Learning Course Forums](http://forums.fast.ai/t/nlp-any-libraries-dictionaries-out-there-for-fixing-common-spelling-errors/16411/43)
* [How to easily do Topic Modeling with LSA, PSLA, LDA & lda2Vec](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)
* [Facebook Open Sources Dataset on NLP and Navigation Every Data Scientist should Download](https://www.analyticsvidhya.com/blog/2018/07/facebook-open-sources-dataset-on-nlp-and-navigation-every-data-scientist-should-download/)

According to [Wikipedia](https://en.wikipedia.org/wiki/Word_embedding):

> Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing \(NLP\) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with much lower dimension.

Further readings:[https://www.quora.com/What-does-the-word-embedding-mean-in-the-context-of-Machine-Learning/answer/Julien-Despois](https://www.quora.com/What-does-the-word-embedding-mean-in-the-context-of-Machine-Learning/answer/Julien-Despois)[https://www.tensorflow.org/tutorials/word2vec\#motivation\_why\_learn\_word\_embeddings](https://www.tensorflow.org/tutorials/word2vec#motivation_why_learn_word_embeddings)[https://www.zhihu.com/question/32275069](https://www.zhihu.com/question/32275069)

[Awesome-Chinese-NLP: A curated list of resources for Chinese NLP 中文自然語言處理相關資料](https://github.com/crownpku/Awesome-Chinese-NLP)

[Natural Language Processing Key Terms, Explained](https://www.kdnuggets.com/2017/02/natural-language-processing-key-terms-explained.html)



* [How can I tokenize a sentence with Python?](https://www.oreilly.com/learning/how-can-i-tokenize-a-sentence-with-python)
* [自然語言處理從入門到進階資代碼資源庫彙總（隨時更新）](https://zhuanlan.zhihu.com/p/28616862)
* [艾伦AI研究院发布AllenNLP：基于PyTorch的NLP工具包](https://www.jiqizhixin.com/articles/2017-09-09-5)
* [Deep Learning for NLP Best Practices](http://ruder.io/deep-learning-nlp-best-practices/)
* [Topic Modelling Financial News with Natural Language Processing](http://mattmurray.net/topic-modelling-financial-news-with-natural-language-processing/)
* [Best Practices for Document Classification with Deep Learning](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)
* [Natural Language Processing in Artificial Intelligence is almost human-level accurate. Worse yet, it gets smart!](https://sigmoidal.io/boosting-your-solutions-with-nlp/)
* [Word vectors for non-NLP data and research people](https://medium.com/towards-data-science/word-vectors-for-non-nlp-data-and-research-people-8d689c692353)
* [Deep Learning for NLP Best Practices](http://ruder.io/deep-learning-nlp-best-practices/)
* [初学者指南：神经网络在自然语言处理中的应用](https://www.jiqizhixin.com/articles/2017-09-17-7)
* [A gentle introduction to Doc2Vec](https://medium.com/towards-data-science/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
* [Word embeddings in 2017: Trends and future directions](http://ruder.io/word-embeddings-2017)
* [Word Embedding in Deep Learning](https://analyticsdefined.com/word-embedding-in-deep-learning/)
* [Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
* [Deep Learning for Natural Language Processing: 2016-2017](https://github.com/oxford-cs-deepnlp-2017/lectures)
* [基于Spark /Tensorflow使用CNN处理NLP的尝试](http://www.jianshu.com/p/1afda7000d8e)
* [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
* [Embedding projector - visualization of high-dimensional data](http://projector.tensorflow.org/)
* [Pytorch implementations of various Deep NLP models in cs-224n\(Stanford Univ\)](https://github.com/DSKSD/DeepNLP-models-Pytorch)
* [Stop Using word2vec](http://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/)
* [让机器像人一样交流：斯坦福李纪为博士毕业论文](https://www.jiqizhixin.com/articles/2017-11-14)
* [Gentle Introduction to Statistical Language Modeling and Neural Language Models](https://machinelearningmastery.com/statistical-language-modeling-and-neural-language-models/)
* [Dan Jurafsky & Chris Manning: Natural Language Processing \(great intro video series\)](https://www.youtube.com/watch?v=nfoudtpBV68&list=PL6397E4B26D00A269)
* [A simple spell checker built from word vectors – Ed Rushton – Medium](https://medium.com/@erushton214/a-simple-spell-checker-built-from-word-vectors-9f28452b6f26)
* [Data Science 101 \(Getting started in NLP\): Tokenization tutorial \| No Free Hunch](http://blog.kaggle.com/2017/08/25/data-science-101-getting-started-in-nlp-tokenization-tutorial/)
* [Vector Representations of Words  \|  TensorFlow](https://www.tensorflow.org/tutorials/word2vec) \(highly recommneded by Jeremey\)
* [NLP — Building a Question Answering model – Towards Data Science](https://towardsdatascience.com/nlp-building-a-question-answering-model-ed0529a68c54)
* [Entity extraction using Deep Learning based on Guillaume Genthial work on NER](https://medium.com/intro-to-artificial-intelligence/entity-extraction-using-deep-learning-8014acac6bb8)
* [Text Classification using machine learning – Nitin Panwar – Technical Lead \(Data Science\), Naukri.com](http://nitin-panwar.github.io/Text-Classification-using-machine-learning/)
* [Unsupervised sentence representation with deep learning](https://blog.myyellowroad.com/unsupervised-sentence-representation-with-deep-learning-104b90079a93)
* [**How to solve 90% of NLP problems: a step-by-step guide**](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e) **\(11.3k clap!\)**
* [Building a FAQ Chatbot in Python – The Future of Information Searching](https://www.analyticsvidhya.com/blog/2018/01/faq-chatbots-the-future-of-information-searching/)
* [Sentiment analysis on Trump's tweets using Python](https://dev.to/rodolfoferro/sentiment-analysis-on-trumpss-tweets-using-python-?)
* [Improving Airbnb Yield Prediction with Text Mining – Towards Data Science](https://towardsdatascience.com/improving-airbnb-yield-prediction-with-text-mining-9472c0181731)
* [Machine Learning with Text in scikit-learn \(PyCon 2016\) - YouTube](https://www.youtube.com/watch?v=ZiKMIuYidY0&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=10)
* [Natural Language Processing Nuggets: Getting Started with NLP](https://www.kdnuggets.com/2018/06/getting-started-natural-language-processing.html)
* [Machine Learning as a Service: Part 1 – Towards Data Science](https://towardsdatascience.com/machine-learning-as-a-service-487e930265b2)
* [Text Generation using a RNN](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/text_generation.ipynb)
* [Text Classification Using CNN, LSTM and Pre-trained Glove Word Embeddings: Part-3](https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa)
* [Ahmed BESBES - Data Science Portfolio – Overview and benchmark of traditional and deep learning models in text classification](https://ahmedbesbes.com/overview-and-benchmark-of-traditional-and-deep-learning-models-in-text-classification.html)
* [The 7 NLP Techniques That Will Change How You Communicate in the Future \(Part I\)](https://heartbeat.fritz.ai/the-7-nlp-techniques-that-will-change-how-you-communicate-in-the-future-part-i-f0114b2f0497)
* [Natural Language Processing: What are algorithms for auto summarize text? - Quora](https://www.quora.com/Natural-Language-Processing-What-are-algorithms-for-auto-summarize-text#)
* [A Practitioner's Guide to Natural Language Processing \(Part I\) — Processing & Understanding Text](https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72)
* [Salesforce has Developed One Single Model to Deal with 10 Different NLP Tasks](https://www.analyticsvidhya.com/blog/2018/06/salesforce-has-developed-one-single-model-to-deal-with-10-different-nlp-tasks/)
* [Samsung's ConZNet Algorithm just won Two Popular NLP Challenges \(Dataset Links Inside\)](https://www.analyticsvidhya.com/blog/2018/07/samsungs-conznet-algorithm-won-2-huge-nlp-competitions/)
* [Detecting Sarcasm with Deep Convolutional Neural Networks](https://medium.com/dair-ai/detecting-sarcasm-with-deep-convolutional-neural-networks-4a0657f79e80)
* [Detecting Emotions with CNN Fusion Models – DAIR – Medium](https://medium.com/dair-ai/detecting-emotions-with-cnn-fusion-models-b066944969c8)
* [What is the best tool to summarize a text document? - Quora](https://www.quora.com/What-is-the-best-tool-to-summarize-a-text-document)
* [Text Classification: Applications and Use Cases - ParallelDots](https://blog.paralleldots.com/product/text-analytics/text-classification-applications-use-cases/)
* [**重磅譯制 \| 更新：牛津大學xDeepMind自然語言處理 第10講 文本轉語音（2） \| 機器之心**](https://www.jiqizhixin.com/articles/2018-07-16-18)\*\*\*\*
* [技術解讀 \| 基於fastText和RNN的語義消歧實戰 \| 機器之心](https://www.jiqizhixin.com/articles/2018-07-05-20)
* [葉志豪：介紹強化學習及其在 NLP 上的應用 \| 分享總結 \| 雷鋒網](https://www.leiphone.com/news/201807/sbyafpzV4BgvjLT1.html)
* [Convolutional neural networks for language tasks - O'Reilly Media](https://www.oreilly.com/ideas/convolutional-neural-networks-for-language-tasks?mkt_tok=eyJpIjoiWm1NNFpEUXpNek00TVRkbSIsInQiOiI1ZWEwejVJeUs4SVFhMFJ4b0F5NkZNRlpsTWlqRVdqcWFta001Mkd0YTQzbHY2Qnl0aDlaRkJEM1FEbGN6Ykx6ejFFNW1UVFZDME5jZmd1VEUyYXBBeFQ3VzhCZkRmVDc3dzF3OE1tUUYyYzI4bVo1Vmg5ZXVaUzFFbkphNEwrYiJ9)
* [A Comprehensive Guide to Understand and Implement Text Classification in Python](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/)
* [Jeremy Howard on Twitter: "Very interesting - combining recent work from @lnsmith613 & @GuggerSylvain on super-convergence with a transformer language model \(@AlecRad\) shows dramatic improvements in perplexity, speed, and size over @Smerity's very strong AWD LSTM! https://t.co/t6LbAKap3M https://t.co/xI9E8zHZP8"](https://mobile.twitter.com/jeremyphoward/status/1020097463376461824)
* [kororo/excelcy: Excel Integration with SpaCy. Includes, Entity training, Entity matcher pipe.](https://github.com/kororo/excelcy)
* [dongjun-Lee/text-classification-models-tf: Tensorflow implementations of Text Classification Models.](https://github.com/dongjun-Lee/text-classification-models-tf)
* [dongjun-Lee/transfer-learning-text-tf: Tensorflow implementation of Semi-supervised Sequence Learning \(https://arxiv.org/abs/1511.01432\)](https://github.com/dongjun-Lee/transfer-learning-text-tf)
* [IndicoDataSolutions/finetune: Scikit-learn style model finetuning for NLP](https://github.com/IndicoDataSolutions/finetune)
* [\[1807.00914\] Modeling Language Variation and Universals: A Survey on Typological Linguistics for Natural Language Processing](https://arxiv.org/abs/1807.00914)
* \*\*\*\*[**Introducing state of the art text classification with universal language models · fast.ai NLP**](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)\*\*\*\*
* [香儂科技獨家對話斯坦福大學計算機學院教授、麥克阿瑟天才獎得主Dan Jurafsky \| 機器之心](https://www.jiqizhixin.com/articles/2018-07-25-4)
* [NLP概述和文本自動分類算法詳解 \| 機器之心](https://www.jiqizhixin.com/articles/2018-07-25-5)
* [Multi-Class Text Classification with Scikit-Learn – Towards Data Science](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)
* [Comparison of Top 6 Python NLP Libraries – ActiveWizards: machine learning company – Medium](https://medium.com/activewizards-machine-learning-company/comparison-of-top-6-python-nlp-libraries-c4ce160237eb)
* [李航教授展望自然語言對話領域：現狀與未來 \| 機器之心](https://www.jiqizhixin.com/articles/2018-07-26-9)
* [The unreasonable effectiveness of one neuron \| Rakesh Chada's Blog](https://rakeshchada.github.io/Sentiment-Neuron.html)
* [在自然語言處理領域，哪些企業的發展遙遙領先？（附報告） - CSDN博客](https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/81229829)
* [ACL 2018：Attention 機制佔主流，中文語法檢錯測評引關注 \| ACL 2018 \| 雷鋒網](https://www.leiphone.com/news/201807/CSiDyhfKmCUMsTHy.html)
* [harvardnlp/var-attn](https://github.com/harvardnlp/var-attn)

### Sentiment Analysis

* [Perform sentiment analysis with LSTMs, using TensorFlow - O'Reilly Media](https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow)
* [Data Science 101: Sentiment Analysis in R Tutorial \| No Free Hunch](http://blog.kaggle.com/2017/10/05/data-science-101-sentiment-analysis-in-r-tutorial/)
* [Sentiment Analysis through LSTMs – Towards Data Science](https://towardsdatascience.com/sentiment-analysis-through-lstms-3d6f9506805c)
* [A Beginner’s Guide on Sentiment Analysis with RNN – Towards Data Science](https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e)
* [Twitter Sentiment Analysis using combined LSTM-CNN Models – B-sides](http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/)

### Text Classification

* [A Comprehensive Guide to Understand and Implement Text Classification in Python](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/)
* [Big Picture Machine Learning: Classifying Text with Neural Networks and TensorFlow](https://medium.freecodecamp.org/big-picture-machine-learning-classifying-text-with-neural-networks-and-tensorflow-d94036ac2274)
* [Step 2.5: Choose a Model \| ML Universal Guides \| Google Developers](https://developers.google.com/machine-learning/guides/text-classification/step-2-5)



[Analyzing tf-idf results in scikit-learn - datawerk](https://buhrmann.github.io/tfidf-analysis.html)



Tf-idf stands for _term frequency-inverse document frequency_

> Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency \(TF\), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency \(IDF\), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
>
> * **TF: Term Frequency**, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length \(aka. the total number of terms in the document\) as a way of normalization:   TF\(t\) = \(Number of times term t appears in a document\) / \(Total number of terms in the document\).
> * **IDF: Inverse Document Frequency**, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:   IDF\(t\) = log\_e\(Total number of documents / Number of documents with term t in it\).
>
> See below for a simple example.
>
> **Example:**
>
> Consider a document containing 100 words wherein the word cat appears 3 times. The term frequency \(i.e., tf\) for cat is then \(3 / 100\) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency \(i.e., idf\) is calculated as log\(10,000,000 / 1,000\) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 \* 4 = 0.12.

## Chinese

*  [Chinese Word Vectors：目前最全的中文預訓練詞向量集合 \| 機器之心](https://www.jiqizhixin.com/articles/2018-05-15-10)
*  [Embedding/Chinese-Word-Vectors: 100+ Chinese Word Vectors 上百種預訓練中文詞向量](https://github.com/Embedding/Chinese-Word-Vectors)

## Text Analysis using Machine Learning

Most of the algorithms accept only numerical feature vectors \(`vector` is a one dimensional `array` in computer science\). So we need to convert the text documents into numerical features vectors with a fixed size in order to make use of the machining learning algorithms for text analysis.

This can be done by the following steps:

1. Assign each of the words in the text documents an integer ID. Each of the words is called a `token`. This step is called `tokenization`.
2. Count the occurrences of tokens in each document. This step is called `counting`. The count of each token is created as a feature.
3. `Normalization` \(**Don't understand what it means at this moment**\)

**\(to add easy-to-understand example\)**

This process is called `vectorization`. The resulting numerical feature vectors is called a `bag-of-words` representation.

One issue of `vectorization` is that longer documents will have higher average count values than shorter documents while they might talk about the same topic. The solution is to divide the number of occurrences of each word in a document by total number of words in the document. These features are called `term frequency` or `tf`.

Another issue `vectorization` is that in a large text corpus the common words like "the", "a", "is" will shadow the rare words during the model induction. The solution is to downscale the weight of the words that appear in many documents. This downscaling is called `term frequency times inverse document frequency` or `tf-idf` .

I learnt the above from a [scikit-learn tutorial](http://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation).

According to [Kaggle](https://www.kaggle.com/c/quora-question-pairs/rules), `word embedding` is an example of `pre-trained models`. The followings are the embeddings mentioned by [Kaggle competitors](https://www.kaggle.com/c/quora-question-pairs/discussion/30286):

* [word2vec by Google](https://code.google.com/archive/p/word2vec/)
* [GloVe](https://nlp.stanford.edu/projects/glove/)
* [fastText by Facebook](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

[Kaggle](https://www.kaggle.com/c/quora-question-pairs/discussion/30286) requires competitors to share the pre-trained models and word embeddings used to "keep the competition fair by making sure that everyone has access to the same data and pretrained models."

What is `pre-trained models`?

What is `word embedding`?

Some other tools:

* [Gensim](https://radimrehurek.com/gensim/)
* [spaCy](https://spacy.io/)
* [Amazon Machine Learning](https://aws.amazon.com/machine-learning/)

 [Google AI Blog: Text summarization with TensorFlow](https://ai.googleblog.com/2016/08/text-summarization-with-tensorflow.html)

* [NervanaSystems/nlp-architect: NLP Architect by Intel AI Lab: Python library for exploring the state-of-the-art deep learning topologies and techniques for natural language processing and natural language understanding](https://github.com/NervanaSystems/nlp-architect)
* [NLTK Book](https://www.nltk.org/book/)
* [NLP in Online Courses: an Overview – sciforce – Medium](https://medium.com/sciforce/nlp-in-online-courses-an-overview-7b60c3aec6fa)
* [Home: AAN](http://tangra.cs.yale.edu/newaan/)
* [A simple spell checker built from word vectors – Noteworthy - The Journal Blog](https://blog.usejournal.com/a-simple-spell-checker-built-from-word-vectors-9f28452b6f26)
* [facebookresearch/fastText: Library for fast text representation and classification.](https://github.com/facebookresearch/fastText#example-use-cases)
* [再谈最小熵原理：“飞象过河”之句模版和语言结构 \| 附开源NLP库 \| 机器之心](https://www.jiqizhixin.com/articles/2018-06-01-10)
* [从无监督构建词库看「最小熵原理」，套路是如何炼成的](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247488802&idx=1&sn=eb35229374ee283d5c54d58ae277b9f0&chksm=96e9caa2a19e43b4f624eac3d56532cb9dc7ca017c9e0eaf96387e20e5f985e37da833fbddfd&scene=21#wechat_redirect)
* [gt-nlp-class/notes at master · jacobeisenstein/gt-nlp-class](https://github.com/jacobeisenstein/gt-nlp-class/tree/master/notes)
* [干货 \| 如何从编码器和解码器两方面改进生成式句子摘要？ \| 机器之心](https://www.jiqizhixin.com/articles/2018-06-04-13)
* [Deep Learning for Conversational AI](https://www.poly-ai.com/naacl18)
* [gt-nlp-class/notes at master · jacobeisenstein/gt-nlp-class](https://github.com/jacobeisenstein/gt-nlp-class/tree/master/notes)
* [📚The Current Best of Universal Word Embeddings and Sentence Embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
* [How To Create a ChatBot With tf-seq2seq For Free! – Deep Learning as I See It](https://blog.kovalevskyi.com/how-to-create-a-chatbot-with-tf-seq2seq-for-free-e876ea99063c)
* [ryanjgallagher.github.io/2018-SICSS-InfoTheoryTextAnalysis-Gallagher.pdf at master · ryanjgallagher/ryanjgallagher.github.io](https://github.com/ryanjgallagher/ryanjgallagher.github.io/blob/master/files/slides/2018-SICSS-InfoTheoryTextAnalysis-Gallagher.pdf)
* [nmt\_with\_attention.ipynb - Colaboratory](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb)
* [minimaxir/textgenrnn: Easily train your own text-generating neural network of any size and complexity on any text dataset with a few lines of code.](https://github.com/minimaxir/textgenrnn)
* [Detecting Sarcasm with Deep Convolutional Neural Networks](https://medium.com/dair-ai/detecting-sarcasm-with-deep-convolutional-neural-networks-4a0657f79e80)
* [plasticityai/magnitude: A fast, efficient universal vector embedding utility package.](https://github.com/plasticityai/magnitude)
* [sebastianruder/NLP-progress: Repository to track the progress in Natural Language Processing \(NLP\), including the datasets and the current state-of-the-art for the most common NLP tasks.](https://github.com/sebastianruder/NLP-progress)
* [HUBOT \| Hubot is your friendly robot sidekick. Install him in your company to dramatically improve employee efficiency.](https://hubot.github.com/)
* [lanwuwei/SPM\_toolkit: Neural network toolkit for sentence pair modeling.](https://github.com/lanwuwei/SPM_toolkit)
* [Holy NLP! Understanding Part of Speech Tags, Dependency Parsing, and Named Entity Recognition • Peter Baumgartner](https://pmbaumgartner.github.io/blog/holy-nlp/)
* [bfelbo/DeepMoji: State-of-the-art deep learning model for analyzing sentiment, emotion, sarcasm etc.](https://github.com/bfelbo/DeepMoji)
* [腾讯AI Lab副主任俞栋：语音识别领域的现状与进展 \| 机器之心](https://www.jiqizhixin.com/articles/2018-06-30-13)
* [Word2Vec — a baby step in Deep Learning but a giant leap towards Natural Language Processing](https://towardsdatascience.com/word2vec-a-baby-step-in-deep-learning-but-a-giant-leap-towards-natural-language-processing-40fe4e8602ba)
* [NervanaSystems/nlp-architect: NLP Architect by Intel AI Lab: Python library for exploring the state-of-the-art deep learning topologies and techniques for natural language processing and natural language understanding](https://github.com/NervanaSystems/nlp-architect)
* [IBM Unveils System That ‘Debates’ With Humans - The New York Times](https://www.nytimes.com/2018/06/18/technology/ibm-debater-artificial-intelligence.html)
* [Kyubyong/nlp\_tasks: Natural Language Processing Tasks and References](https://github.com/Kyubyong/nlp_tasks)
* [nateraw/Lda2vec-Tensorflow: Tensorflow 1.5 implementation of Chris Moody's Lda2vec, adapted from @meereeum](https://github.com/nateraw/Lda2vec-Tensorflow)
* [LDA2vec: Word Embeddings in Topic Models \(article\) - DataCamp](https://www.datacamp.com/community/tutorials/lda2vec-topic-model)
* [cemoody/lda2vec](https://github.com/cemoody/lda2vec)
* [The Natural Language Decathlon](https://einstein.ai/research/the-natural-language-decathlon)
* [🚀 100 Times Faster Natural Language Processing in Python](https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced)
* [How to easily do Topic Modeling with LSA, PSLA, LDA & lda2Vec](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)
* [Deep-learning-free Text and Sentence Embedding, Part 2 – Off the convex path](http://www.offconvex.org/2018/06/25/textembeddings/)
* [Answering English questions using knowledge graphs and sequence translation](https://medium.com/octavian-ai/answering-english-questions-using-knowledge-graphs-and-sequence-translation-2acbaa35a21d)
* [Ahmed BESBES - Data Science Portfolio – Overview and benchmark of traditional and deep learning models in text classification](https://ahmedbesbes.com/overview-and-benchmark-of-traditional-and-deep-learning-models-in-text-classification.html)f
* [A Practitioner's Guide to Natural Language Processing \(Part I\) — Processing & Understanding Text](https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72)
* [LDA2vec: Word Embeddings in Topic Models – Towards Data Science](https://towardsdatascience.com/lda2vec-word-embeddings-in-topic-models-4ee3fc4b2843)
* [Multi-Class Text Classification with Scikit-Learn – Towards Data Science](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)
* [Rasa: Open source conversational AI](https://rasa.com/)
* [Deep Learning for Natural Language Processing: Tutorials with Jupyter Notebooks](https://insights.untapt.com/deep-learning-for-natural-language-processing-tutorials-with-jupyter-notebooks-ad67f336ce3f)
* [Keras LSTM tutorial - How to easily build a powerful deep learning language model - Adventures in Machine Learning](http://adventuresinmachinelearning.com/keras-lstm-tutorial/)
* [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
* [Transfer Learning for Text using Deep Learning Virtual Machine \(DLVM\) \| Machine Learning Blog](https://blogs.technet.microsoft.com/machinelearning/2018/04/25/transfer-learning-for-text-using-deep-learning-virtual-machine-dlvm/)
* [Ahmed BESBES - Data Science Portfolio – Overview and benchmark of traditional and deep learning models in text classification](https://ahmedbesbes.com/overview-and-benchmark-of-traditional-and-deep-learning-models-in-text-classification.html)
* [**NLP's ImageNet moment has arrived**](https://thegradient.pub/nlp-imagenet/)
* [Generating Text with RNNs in 4 Lines of Code](https://www.kdnuggets.com/2018/06/generating-text-rnn-4-lines-code.html)
* [全球首家多語言智能客服上線，這家神秘AI公司有什麼秘密武器？ \| 機器之心](https://www.jiqizhixin.com/articles/2018-07-22-2)
* [Transfer Learning in Natural Language Processing \| Intel® Software](https://software.intel.com/en-us/articles/transfer-learning-in-natural-language-processing)
* [dmlc/gluon-nlp: NLP made easy](https://github.com/dmlc/gluon-nlp)
* [Introduction \| ML Universal Guides \| Google Developers](https://developers.google.com/machine-learning/guides/text-classification/?linkId=54683504)
* \*\*\*\*[**AutoML Natural Language Beginner's guide \| AutoML Natural Language \| Google Cloud**](https://cloud.google.com/natural-language/automl/docs/beginners-guide)\*\*\*\*

