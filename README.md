# Portfolio

## Stock Price Prediction
An investigation into whether there is any statistically detectable signal in stock market data to predict next-day directional change using walk-forward validation and Pesaran-Timmermann test. Common statistical properties of stock data are examined and confirmed via various statistical tests: excess kurtosis in returns, non-stationarity, volatility clustering. An ARIMA model was fitted to the data using the AIC criterion and its effectivness assessed via walk-forward validation. The model was updated every 20 trading days. Directional accuracy in stock returns was 54.99% with a Pesaran-Timmermann test yielding a p-value of 0.032.

[View project](./Stock-Price-Prediction/)

## ML Algorithms
Implementation of standard ML algorithms in Python using only NumPy and pandas with mathematical derivations. Sklearn models are used as a baseline comparison for performance of ML algorithms. Algorithms implemented include: OLS, ridge and lasso regression, logistic regression, KNN, all of which achieved a perfect comparison with sklearn models.

[View project](./ML-Algorithms/)

## Options Pricing
A derivation of the analytical solution to the Black-Scholes equation for pricing a European option is provided. Numerical solutions in the case of non-dividend paying American options are provided using both binomial tree and finite difference methods. The finite difference scheme employs a modified Thomas algorithm in order to impose the early-exercise constraint. The numerical methods are successfully benchmarked against known analytic solutions and properties. *This project is currently ongoing and more will be added.*

[View project](./Options-Pricing/)
