import numpy as np

def coefficients(X, y):
    '''
    Computes the OLS regression coefficients

    INPUTS:
    X - N x D matrix of training inputs
    y - N x 1 vector of training targets/observations

    OUTPUS:
    OLS coefficients (D x 1) array
    '''
    
    beta = np.linalg.pinv(X) @ y
    return beta

def predict(X, beta):
    '''
    Returns the predicted values based on OLS regression.

    INPUTS:
    X       N x D matrix of training or test data
    beta    D x 1 array of estimated coefficients

    OUTPUTS:
    y_pred  N x 1 array of predicted values 
    '''

    y_pred = X @ beta
    return y_pred