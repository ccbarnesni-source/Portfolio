## Lasso Regression

Standard Lasso regression aims to reduce variance in linear models by introduces an $l_1$ penalty. A notable advantage of Lasso regression is its ability to perform feature selection. With the addition of the penality term, the optimisation problem can be formulated as:
$$
\min_{\boldsymbol{\beta}} \bigg( ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta} ||^2_2 + \lambda || \boldsymbol{\beta} ||_1 \bigg),
$$
where $|| \boldsymbol{\beta} ||_1 = \sum_{i=0}^n |\beta_i| $.
If we let $L(\boldsymbol{\beta})$ denote the loss function above which we are trying to minimise, then it is easy to see that the gradient is
$$
\nabla_{\boldsymbol{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + \lambda \mathbf{sgn}(\boldsymbol{\beta}).
$$
Here, we have defined $\mathbf{sgn}$ to be the vector generalisation of the sign function so that $\mathbf{sgn}(\mathbf{x})=\text{sgn}(x_i)\mathbf{e}_i$, i.e., it returns the sign of each component of the vector. From here, we can employ gradient descent methods to find the desired coefficients.