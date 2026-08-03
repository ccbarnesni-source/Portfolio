import numpy as np
import pandas as pd

def standardise(X):
    '''
    Standardises a dataset, i.e. returns (X - mu) / sigma

    INPUTS:
    X - an n x p array or dataframe consisting of n observations with p predictors

    OUTPUT:
    the standardised version of that array
    '''
    
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_std = (X-mu)/(sigma)
    return X_std