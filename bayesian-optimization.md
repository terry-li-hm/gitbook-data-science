# Bayesian Optimization

[Ax](https://ax.dev/docs/bayesopt.html):

> In complex engineering problems we often come across parameters that have to be tuned using several time-consuming and noisy evaluations. When the number of parameters is not small or if some of the parameters are continuous, using large factorial designs \(e.g., “grid search”\) or global optimization techniques for optimization require too many evaluations than is practically feasible. These types of problems show up in a diversity of applications, such as
>
> 1. Tuning Internet service parameters and selection of weights for recommender systems,
> 2. **Hyperparameter optimization for machine learning,**
> 3. Finding optimal set of gait parameters for locomotive control in robotics,
> 4. Tuning design parameters and rule-of-thumb heuristics for hardware design.
>
> Bayesian optimization \(BO\) allows us to tune parameters in relatively few iterations by building a smooth model from an initial set of parameter configurations \(referred to as the "surrogate model"\) to predict the outcomes for yet unexplored parameter configurations. This represents an adaptive approach where the observations from previous evaluations are used to decide what parameter configurations to evaluate next. The same strategy can be used to predict the expected gain from all future evaluations and decide on early termination, if the expected benefit is smaller than what is worthwhile for the problem at hand.

