# Tree

* [kjw0612/awesome-random-forest: Random Forest - a curated list of resources regarding random forest](https://github.com/kjw0612/awesome-random-forest)
* [Introduction to Decision Tree Learning – Heartbeat](https://heartbeat.fritz.ai/introduction-to-decision-tree-learning-cd604f85e236)
* [Quick Guide to Boosting Algorithms in Machine Learning](https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/)
* [Introduction to Random Forest - All About Analytics](https://analyticsdefined.com/introduction-random-forests/)
* [Improving the Random Forest in Python Part 1 – Towards Data Science](https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd)
* [Random Forest Simple Explanation – William Koehrsen – Medium](https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d)
* [Random Forest in Python – Towards Data Science](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0)
* [Improving the Random Forest in Python Part 1 – Towards Data Science](https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd)
* [Hyperparameter Tuning the Random Forest in Python – Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
* [Explaining model's predictions \| Kaggle](https://www.kaggle.com/alijs1/explaining-model-s-predictions/notebook)

A random forest is a bunch of independent decision trees each contributing a “vote” to an prediction. E.g. if there are 50 trees, and 32 say “rainy” and 18 say “sunny”, then the score for “rainy” is 32/50, or 64,% and the score for a “sunny” is 18/50, or 36%. Since 64% &gt; 36%, the forest has voted that they think it will rain.  
  
When you add more decision trees to a random forest, they decide what they think INDEPENDENTLY of all the other trees. They learn on their own, and when it comes time to make a prediction, they all just throw their own uninfluenced opinion into the pot.  
  
A gradient boosting model is a CHAIN of decision trees that also each make a vote. But instead of each learning in isolation, when you add a new one to the chain, it tries to improve a bit on what the rest of the chain already thinks. So, a new tree’s decision IS influenced by all the trees that have already voiced an opinion.  
  
Unlike a Random Forest, when you add a new tree to a GBM, it gets to see what its predecessors thought - and how they got it right or wrong. They then formulate a suggestion to correct the errors of their predecessors - and then they add that to the pot, and then the process continues with the next tree you add to the chain.



## Gradient boosting

Gradient boosting is a type of boosting. It relies on the intuition that the best possible next model, when combined with previous models, minimizes the overall prediction error. The key idea is to set the target outcomes for this next model in order to minimize the error. How are the targets calculated? The target outcome for each case in the data depends on how much changing that case’s prediction impacts the overall prediction error:  
  
If a small change in the prediction for a case causes a large drop in error, then next target outcome of the case is a high value. Predictions from the new model that are close to its targets will reduce the error.  
If a small change in the prediction for a case causes no change in error, then next target outcome of the case is zero. Changing this prediction does not decrease the error.  
The name gradient boosting arises because target outcomes for each case are set based on the gradient of the error with respect to the prediction. Each new model takes a step in the direction that minimizes prediction error, in the space of possible predictions for each training case.



## Ensembles and boosting

Machine learning models can be fitted to data individually, or combined in an ensemble. An ensemble is a combination of simple individual models that together create a more powerful new model.  
  
Boosting is a method for creating an ensemble. It starts by fitting an initial model \(e.g. a tree or linear regression\) to the data. Then a second model is built that focuses on accurately predicting the cases where the first model performs poorly. The combination of these two models is expected to be better than either model alone. Repeat the process many times. Each successive model attempts to correct for the shortcomings of the combined boosted ensemble of all previous models.

