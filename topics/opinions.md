# Opinions

[5 Reasons “Logistic Regression” should be the first thing you learn when becoming a Data Scientist](https://towardsdatascience.com/5-reasons-logistic-regression-should-be-the-first-thing-you-learn-when-become-a-data-scientist-fcaae46605c4)



## JPM

### Deep Learning vs. Classical Machine Learning

Despite such successes, Deep Learning tools are rarely used in time series analysis, where tools from ‘classical’ Machine Learning dominate.

Anecdotal evidence from observing winning entries at data science competitions \(like Kaggle\) suggests that structured data is best analyzed by tools like XGBoost and Random Forests. Use of Deep Learning in winning entries is limited to analysis of images or text. Deep Learning tools still require a substantial amount of data to train. Training on small sample sizes \(through so-called generative-adversarial models\) is still at an incipient research stage. The necessity of having large sample data implies that one may see application of Deep Learning to intraday or high-frequency trading before we see its application in lower frequencies.

Deep Learning finds immediate use for portfolio managers in an indirect manner. Parking lot images are analyzed using Deep Learning architectures \(like convolutional neural nets\) to count cars. Text in social media is analyzed using Deep Learning architectures \(like long short-term memory\) to detect sentiment. Such traffic and sentiment signals can be integrated directly into quantitative strategies, as shown in earlier sections of this report. Calculation of such signals themselves will be outsourced to specialized firms that will design bespoke neural network architecture for the task.

### Time-Series Analysis: Long Short-Term Memory

While LSTM is designed keeping in mind such time-series, there is little research available on its application to econometrics or financial time-series. After describing the basic architecture of an LSTM network below, we also provide a preliminary example of a potential use of LSTM in forecasting asset prices.

We attempted to use an LSTM neural network to design an S&P 500 trading strategy. The dataset included S&P 500 returns since 2000, and the first 14 years were used for training the LSTM. The last 3 years worth of data were used to predict monthly returns.

