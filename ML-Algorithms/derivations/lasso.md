## Lasso Regression

Standard Lasso regression aims to reduce variance in linear models by introducing a $\ell_1$ penalty. A notable advantage of Lasso regression is its ability to perform feature selection. With the addition of the penalty term, the optimisation problem can be formulated as:

$$
\min_{\boldsymbol{\beta}} \bigg( \lVert\mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2_2 + \lambda \lVert \boldsymbol{\beta} \rVert_1 \bigg),
$$

where $\lVert {\beta} \rVert_ 1 = \sum_{i=0}^n \lVert\beta_i\rVert_1$.
If we let $L(\boldsymbol{\beta})$ denote the loss function above which we are trying to minimise, then it is easy to see that the gradient is

$$
-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + \lambda \mathbf{sgn}(\boldsymbol{\beta}).
$$

Here, we have defined $\mathbf{sgn}$ to be the vector generalisation of the sign function so that $\mathbf{sgn}(\mathbf{x})=\text{sgn}(x_i)\mathbf{e}_i$, i.e., it returns the sign of each component of the vector. From here, we can employ gradient descent methods to find the desired coefficients.
