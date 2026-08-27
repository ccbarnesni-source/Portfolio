## Logistic Regression

In this section we will fit a logistic regression model. Logistic regression works in a similar manner to linear regression with a few notable extensions. For a 2-class classification problem, we use a linear model to predict the log odds of an observation belonging to the target class. Thus, if $p$ is the probability of our observation belonging to the target class, then we fit a model of the form

$$
\sum_{j=0}^d \beta_j x_{ij} = \log \Big( \frac{p_i}{1-p_i} \Big)
$$

Rearranging, we see that $p$ is in fact the resulting transformation after applying the logistic function to our log odds:

$$
p_i = \frac{e^{\sum_{j=0}^d \beta_j X_j}}{1+e^{\sum_{j=0}^d \beta_j X_j}}
$$

This results in a value bounded by 0 and 1 allowing us to interpret it as a probability. If the probability above exceeds a given threshold (a hyperparameter that we will tune later), we classify it as belonging to the target (in this case, malignant) category.

We may view each observation as an observation of a Bernoulli random variable with probability as given above. We can optimise the parameters by maximising the likelihood function then which, in this case, is

$$
\mathcal{L}(y_i ; \boldsymbol{\beta}) = \prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i}
$$

We now introduce the notation $E_i = e^{\sum_{j=0}^d \beta_j X_j}$ for convenience. Then, the log-likelihood of the above is

$$
\begin{align*}
\log \mathcal{L}(y_i ; \boldsymbol{\beta}) &= \sum_{i=1}^n y_i \log \Big( \frac{E_i}{1+E_i} \Big) + (1-y_i)\log \Big( \frac{1}{1+E_i} \Big) \\
&= \sum_{i=1}^n y_i \log E_i - \log (1+E_i)
\end{align*}
$$

Computing the gradient, we see that

$$
\begin{align*}
\nabla_{\boldsymbol{\beta}}(\log \mathcal{L}(y_i ; \boldsymbol{\beta})) &= \sum_{i=1}^n y_i \mathbf{x}_i - \frac{E_i}{1+E_i}\mathbf{x}_i \\
& = \sum_{i=1}^n \big(y_i - p_i \big) \mathbf{x}_i
\end{align*}
$$

Thus solving $\sum_{i=1}^n \big(y_i - p_i \big) \mathbf{x}_i = \mathbf{0}$ via gradient descent methods will obtain us our logistic model parameters.
