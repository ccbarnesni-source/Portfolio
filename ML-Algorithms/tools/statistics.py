import numpy as np
import pandas as pd

def R_squared(y, y_pred):
    '''
    Computes the R^2 statistic

    INPUTS:
    y - an n x 1 array of true values
    y_pred - an n x 1 array of predicted values from a model

    OUPUTS:
    R - the R^2 statistic
    '''
    
    y_bar = np.mean(y)
    SST = np.sum((y-y_bar)**2)    # Sum of squares total, i.e. the variance of the target variable
    SSE = np.sum((y-y_pred)**2)   # Sum of squared errors
    R = 1 - SSE/SST
    return R

def MSE(y, y_pred):
    '''
    INPUTS:
    y - the observed response
    y_pred - the predicted response from the model

    OUTPUTS:
    MSE - the mean squared error of the prediction
    '''
    
    MSE = np.sum((y-y_pred)**2)/len(y)
    return MSE