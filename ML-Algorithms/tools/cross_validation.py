import numpy as np
import pandas as pd
import models.lasso
import models.ridge
import tools.statistics

def folds_score(X, y, folds, lam, model_type):
    '''
    Employs cross validations on a dataset to obtain the average MSE for ridge or lasso regression.
    Careful with inputs being numpy/pandas and intercept terms being present or not.

    INPUTS:
    X           Matrix of training data of predictor variables
    y           Vector of training data on response variable
    folds       Indices containing each of the folds to implement cross-validation
    lam         A choice of the tuning parameter lambda in ridge regression
    model_type  A string, either 'ridge' or 'lasso'

    OUTPUTS:
    MSE         The average mean squared error on the data given the choice of lambda across the folds
    '''

    if model_type not in ('ridge', 'lasso'):
          raise ValueError(f'Model type must be either \'ridge\' or \'lasso\', got {model_type}')

    # Initialise variables
    score = 0
    N = len(folds)

    # Iterate over the number of folds
    
    for i in range(N):

        # Begin by obtaining the indicies for a given fold and select the relevant observations in the dataset
        
        val_indexes = folds[i]
        X_val = X.iloc[val_indexes]
        y_val = y[val_indexes]
        X_train = X.drop(val_indexes, axis=0)
        y_train = y.drop(val_indexes, axis=0)

        # Obtain the ridge/lasso parameters on the training data

        if model_type == 'ridge':
              beta = models.ridge.coefficients(X_train, y_train, lam)

        else:
             beta = models.lasso.coefficients(X_train, y_train, lam)

        # Obtain the prediction and add the MSE for a single fold to a running tally over all the folds
        
        y_pred = np.array(X_val) @ beta
        score += tools.statistics.MSE(y_val, y_pred)

    # Average the MSE from each fold
        
    score = score/N
    return score

def folds_lasso_score(X, y, folds, lam):
    '''
    Employs cross validations on a dataset to obtain the average MSE for lasso regression
    
    INPUTS:
    X - an n x p array of observations
    y - an n x 1 array of corresponding targets
    folds - a list of lists of indices containing corresponding to the folds to use
    lam - the lasso penalty hyperparameter

    OUTPUS:
    The averaged MSE across each of the folds
    '''
    
    score = 0
    N = len(folds)
    
    for i in range(N):
            val_indexes = folds[i]
            train_indexes = list(set(range(y.shape[0])) - set(val_indexes))
            X_train = X[train_indexes, :]
            y_train = y[train_indexes]
            X_val = X[val_indexes, :]
            y_val = y[val_indexes]
            
            # Add an intercept term
            
            p, q = X_train.shape[0], X_val.shape[0]
            
            # Obtain the lasso parameters on the training data
    
            beta = minimize_lasso(X_train, y_train, lam)
    
            # Obtain prediction
    
            y_pred = X_val @ beta
            fold_MSE = MSE(y_val, y_pred)
            score += fold_MSE
        
    score = score/N
    return score