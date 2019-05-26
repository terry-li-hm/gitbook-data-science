# EDA

## Tools

* [**facets**](https://github.com/PAIR-code/facets)\*\*\*\*



Python libraries represent missing numbers as nan which is short for "not a number". You can detect which cells have missing values, and then count how many there are in each column with the command:

```python
print(data.isnull().sum())
```



## Number of unique values in each column

```python
unique_counts = pd.DataFrame.from_records([(col, df[col].nunique()) for col in df.columns],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
```





[An Introduction to VisiData — An Introduction to VisiData](https://jsvine.github.io/intro-to-visidata/)

[ZhengyaoJiang/PGPortfolio: PGPortfolio: Policy Gradient Portfolio, the source code of "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"\(https://arxiv.org/pdf/1706.10059.pdf\).](https://github.com/ZhengyaoJiang/PGPortfolio)

[https://www.dropbox.com/s/ayfg33w3cgn5hjl/Think%20Stats%20-%20Exploratory%20Data%20Analysis%20in%20Python.pdf?dl=0](https://www.dropbox.com/s/ayfg33w3cgn5hjl/Think%20Stats%20-%20Exploratory%20Data%20Analysis%20in%20Python.pdf?dl=0)

[scipy/scipy: Scipy library main repository](https://github.com/scipy/scipy)

[statsmodels/statsmodels: Statsmodels: statistical modeling and econometrics in Python](https://github.com/statsmodels/statsmodels)

[Python Data Visualizations \| Kaggle](https://www.kaggle.com/benhamner/python-data-visualizations)

> Some common practices:
>
> * Inspect the distribution of target variable. Depending on what scoring metric is used, **an** [**imbalanced**](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5128907) **distribution of target variable might harm the model’s performance**.
> * For **numerical variables**, use **box plot** and **scatter plot** to inspect their distributions and check for outliers.
> * For classification tasks, plot the data with points colored according to their labels. This can help with feature engineering.
> * Make pairwise distribution plots and examine their correlations.

[How to Approach Data: Tabular Data – Apteo – Medium](https://medium.com/apteo/how-to-approach-data-tabular-data-326c94f0f274)

[HVF/franchise: 🍟 a notebook sql client. what you get when have a lot of sequels.](https://github.com/hvf/franchise)

[Python Data Visualizations \| Kaggle](https://www.kaggle.com/benhamner/python-data-visualizations)

* [Analyzing Ethereum, Bitcoin, and 1200+ other Cryptocurrencies using PostgreSQL](https://blog.timescale.com/analyzing-ethereum-bitcoin-and-1200-cryptocurrencies-using-postgresql-3958b3662e51)
* [The Next Wave: Predicting the future of coffee in New York City](https://medium.com/topos-ai/the-next-wave-predicting-the-future-of-coffee-in-new-york-city-23a0c5d62000)
* [Humble Intro to Analysis with Pandas and Seaborn](https://www.kaggle.com/crawford/humble-intro-to-analysis-with-pandas-and-seaborn/)
* [This Is Where Hate Crimes Don’t Get Reported](http://projects.propublica.org/graphics/hatecrime-map)



* Plot a word cloud to find out what are the most common words.
* Semantic Analysis: Check the usage of different punctuations in dataset.



When handling image classification problems, try to answer the following questions:

* What are the distributions of image types?
* Are the images in the same dimension?

[How to Use Correlation to Understand the Relationship Between Variables](https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/)



* [quantopian/qgrid: An interactive grid for sorting, filtering, and editing DataFrames in Jupyter notebooks](https://github.com/quantopian/qgrid)
* [A Complete Machine Learning Walk-Through in Python: Part One](https://towardsdatascience.com/a-complete-machine-learning-walk-through-in-python-part-one-c62152f39420)
* [ResidentMario/missingno: Missing data visualization module for Python.](https://github.com/ResidentMario/missingno)
* [Are lego sets too pricey ? \| Kaggle](https://www.kaggle.com/jonathanbouchet/are-lego-sets-too-pricey)
* [Interactive maps and EDA of gun violence in the US \| Kaggle](https://www.kaggle.com/erikbruin/interactive-maps-and-eda-of-gun-violence-in-the-us/notebook)
* [Insightful & Vast USA Statistics EDA & EFA \| Kaggle](https://www.kaggle.com/alexgeiger/insightful-vast-usa-statistics-eda-efa/notebook)
* [DIVE \| Turn Data into Stories Without Writing Code](https://dive.media.mit.edu/)
* [Happiness 2017 \(Visualization + Prediction\) \| Kaggle](https://www.kaggle.com/javadzabihi/happiness-2017-visualization-prediction/notebook)
* [Profiling Top Kagglers: Martin Henze \(AKA Heads or Tails\), World’s First Kernels Grandmaster \| No Free Hunch](http://blog.kaggle.com/2018/06/19/tales-from-my-first-year-inside-the-head-of-a-recent-kaggle-addict/)
* [Manning \| Exploring Data with Python](https://www.manning.com/books/exploring-data-with-python)
* [DIVE - MIT's Open Source Tool for Data Exploration and Visualization for Data Scientists](https://www.analyticsvidhya.com/blog/2018/06/perform-data-exploration-with-a-single-click-dive/)
* [Stock Data Analysis with Python \(Second Edition\) \| Curtis Miller's Personal Website](https://ntguardian.wordpress.com/2018/07/17/stock-data-analysis-python-v2/)
* [Google AI Blog: Facets: An Open Source Visualization Tool for Machine Learning Training Data](https://ai.googleblog.com/2017/07/facets-open-source-visualization-tool.html)
* [Data Exploration with Python, Part 1 — District Data Labs: Data Science Consulting and Training](https://www.districtdatalabs.com/data-exploration-with-python-1)
* [Data Exploration with Python, Part 2 — District Data Labs: Data Science Consulting and Training](https://www.districtdatalabs.com/data-exploration-with-python-2)
* [Data Exploration with Python, Part 3 — District Data Labs: Data Science Consulting and Training](https://www.districtdatalabs.com/data-exploration-with-python-3)
* [A Complete Tutorial which teaches Data Exploration in detail](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/)

