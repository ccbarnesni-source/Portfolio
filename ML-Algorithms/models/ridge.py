import numpy as np
import pandas as pd

def coefficients(X,y,lam):
    '''
    Computes the ridge regression coefficients

    INPUTS:
    X - an n x p array of observations
    y - an n x 1 array of corresponding observations
    lam - the hyperparameter corresponding to the penalty term

    OUTPUTS:
    an p x 1 array of ridge regression coefficients
    '''

    N, M = X.shape
    
    # Calculate the parameters
    
    I = np.identity(M)
    beta = (np.linalg.inv(X.T @ X + lam * I)) @ X.T @ y
    
    return beta

def predict(X, beta):
    '''
    Returns the predicted values based on ridge regression.

    INPUTS:
    X       N x D matrix of training or test data
    beta    D x 1 array of estimated coefficients

    OUTPUTS:
    y_pred  N x 1 array of predicted values 
    '''

    y_pred = X @ beta
    return y_pred