## Ridge Regression

Ridge regression is the same as OLS regression but with the addition of a penalty term (given by an $l_2$ norm) on the coefficients. This thus acts to shrink the coefficients. Naturally, this increases the bias of the model but it also decreases the variance. Thus, controlling the multiple of the penalty term allows us to tune our model to avoid overfitting; we have no such control in OLS regression. The ridge regression coefficients are thus the solution to the optimisation problem
$$
\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta} ||^2_2 + \lambda || \boldsymbol{\beta }||_2,
$$
where $\lambda$ is a hyperparameter to be tuned later. The problem is solved similarly as to before by computing the gradient. We need only add an additional term due to the penalty function:
$$
\nabla_{\boldsymbol{\beta}}f = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}  + 2\lambda\boldsymbol{\beta}.
$$
We see that this is $\mathbf{0}$ when
$$
\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X}+ \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}.
$$
Similarly, the Hessian in this case is just
$$
\nabla^2_{\boldsymbol{\beta}}f = \mathbf{X}^T\mathbf{X} + 2\lambda \mathbf{I},
$$
which is also positive definite assuming both $\mathbf{X}\neq\mathbf{0}$ and $\lambda>0$.