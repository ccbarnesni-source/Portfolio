import numpy as np
import pandas as pd

def grad_l1(beta):
    '''
    Computes the vector generalisation of the sgn function (as defined in markdown derivation file)

    INPUTS:
    beta - a p x 1 array of coefficients

    OUPUTS:
    p x 1 array corresponding to the gradient of the input
    '''
    
    p = len(beta)
    grad = np.zeros(p)
    
    for i in range(p):
        if beta[i] < 0 :
            grad[i] = -1
        elif beta[i] > 0 :
            grad[i] = 1
        else:
            grad[i] = 0
    return grad

assert np.all(np.equal( grad_l1(np.array([5,1,-1,0])),  np.array([1,1,-1,0]) ))

def coefficients(X, y, lam, max_iters=10000, learning_rate=1e-3, tol=1e-8):
    '''
    Computes the lasso regression coefficients using gradient descent methods

    INPUTS:
    X - an n x p array of observations
    y - an n x 1 array of corresponding targets
    lam - the lasso penalty hyperparameter
    n_iters - the number of gradient descent iterations to use
    step_size - the size of each step in the gradient descent algorithm

    OUTPUTS:
    beta - a p x 1 array of lasso regression coefficients
    '''

    # Check hyperparameter is valid
    if lam<0:
        raise ValueError(f'Must have lam>=0, got {lam}.')

    # If pandas series/dataframe is given, convert to a numpy array first
    X = np.asarray(X)
    y = np.asarray(y)

    n, p = X.shape
    XX = X.T @ X  
    Xy = X.T @ y 
    
    # Initialise betas
    beta = np.zeros(p)

    # Gradient descent

    prev_beta = np.inf * np.ones(p)
    converged = False

    for i in range(max_iters):
        grad = -2*Xy + 2*XX @ beta + lam*grad_l1(beta)
        
        # Gradient descent update
        beta = beta - grad*learning_rate

        if np.sqrt(np.sum((beta - prev_beta)**2)) < tol:
            print('Converged before max iterations')
            converged = True
            break
    if not converged:
        print('Failed to converge before max iterations reached')

    return beta

def predict(X, beta):
    '''
    Returns the predicted values based on Lasso regression.

    INPUTS:
    X       N x D matrix of training or test data
    beta    D x 1 array of estimated coefficients

    OUTPUTS:
    y_pred  N x 1 array of predicted values 
    '''

    y_pred = X @ beta
    return y_pred