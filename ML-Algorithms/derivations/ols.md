## Ordinary Least Squares Regression

Linear regression is the simplest machine learning model there is. It begins with a dataset of observations, each observation being some linear combination of predictors. We will represent this by an $n \times p$ matrix $\textbf{X}$ for $n$ observations and $p$ predictors. We then have a target variable $\mathbf{y}$ which is a $n$-vector of corresponding targets to our dataset. The goal is to predict some new observation $y$ from an unseen observation $\mathbf{x}_i$. In practice, this new observation will come from the test set of our data. To this end we make a prediction, $\mathbf{\hat{y}}$, of the form $\mathbf{\hat{y}}=\mathbf{X}\boldsymbol{\beta}$ for some vector of parameters $\boldsymbol{\beta}$. Thus, a single prediction can be written as 
$$
\hat{y}_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip}
$$
Our predictions thus produce errors which we call residuals (defined by $y_i - \hat{y}_i$) and we seek to minimise the RSS (residual sum of squares)  these. We can thus formulate our problem as the optimisation problem
$$
\min_{\boldsymbol{\beta}} || \mathbf{y} - \mathbf{X}\boldsymbol{\beta} ||^2
$$
Computing the gradient we have
$$
\nabla_{\boldsymbol{\beta}} f(\boldsymbol{\beta}) = -2\mathbf{X}^T(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})
$$
where here we have set $f(\mathbf{X}) = || \mathbf{y} - \mathbf{X}\boldsymbol{\beta} ||^2 $. Now setting the gradient to $\mathbf{0}$ we have
$$ \boldsymbol{\beta} = \big( \mathbf{X}^T \mathbf{X} \big)^{-1} \mathbf{X}^T\mathbf{y} $$
Note that this can simplify to
$$
\boldsymbol{\beta} = \mathbf{X}^{-1} \mathbf{y}
$$
which can be computed directly only if $\mathbf{X}^{-1}$ exists. This is typically not the case. If it does not, we must compute the full expression above. We can verify that this is indeed a minimum by examining the Hessian,
$$
\nabla^2_{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{X}
$$
which is positive-definite assuming $\mathbf{X}\neq\mathbf{0}$. Thus our answer is indeed a minimum.