## KNN Classifier

The premise of a kNN Classifier is simple. Let our training data consist of $n$ observations of $p$ predictors via the matrix $\mathbf{X}$. Suppose each observation has a corresponding binary classification given by the vector $\mathbf{y}$. Then, for some new observation, $x^T$, we select the $k$ closest observations from $X$ and compute the modal class amongst those $k$ (or mean for regression tasks). The idea of closeness can use any metric, although we typically use the $l_2$ norm, i.e.

$$
|| \mathbf{a} - \mathbf{b}||_2 = \sum_{i=1}^p (a_i-b_i)^2
$$
