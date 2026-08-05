import numpy as np
import pandas as pd
import models.lasso
import models.ridge
import models.knn
import models.logistic
import tools.statistics

def cross_val_score(X, y, folds, model_type, lam=None, k=None):
    '''
    Employs cross validations on a dataset to obtain the average MSE for ridge/lasso regression or the average accuracy/recall/precision/f1 score for KNN/logistic regression.

    INPUTS:
    X           Matrix of training data of predictor variables
    y           Vector of training data on response variable
    folds       Indices containing each of the folds to implement cross-validation
    model_type  A string, either 'ridge', 'lasso', 'knn' or 'logistic'
    lam         A choice of the tuning parameter lambda in ridge regression
    k           The number of neighbours when training the KNN model

    OUTPUTS:
    MSE         The average mean squared error on the data given the choice of lambda across the folds
    '''

    # If pandas series/dataframe is given, convert to a numpy array first
    X = np.asarray(X)
    y = np.asarray(y)

    # Initialise variables
    scores = []
    N = len(folds)

    # Iterate over the number of folds
    for i in range(N):
        
        # Begin by obtaining the indicies for a given fold and select the relevant observations in the dataset
        
        val_indices = folds[i]
        train_indices = list(set(np.arange(len(y))) - set(val_indices))
        X_val = X[val_indices,:]
        y_val = y[val_indices]
        X_train = X[train_indices,:]
        y_train = y[train_indices]

        # Train the models and evaluate on the given fold

        if model_type == 'ridge':
            beta = models.ridge.coefficients(X_train, y_train, lam) # Compute coefficients
            y_pred = models.ridge.predict(X_val, beta) # Compute predicted values
            scores.append(tools.statistics.MSE(y_val, y_pred)) # Compute MSE score

        elif model_type == 'lasso':
            beta = models.lasso.coefficients(X_train, y_train, lam) # Compute coefficients
            y_pred = models.lasso.predict(X_val, beta) # Compute predicted values
            scores.append(tools.statistics.MSE(y_val, y_pred)) # Compute MSE score

        elif model_type == 'knn':
            y_pred = models.knn.predict(X_train, y_train, X_val, k=k) # Compute predicted values
            score_i = tools.statistics.classification_score(y_val, y_pred) # Compute classification scores
            scores.append(score_i)

        elif model_type == 'logistic':
            return 'Under construction! :-)'

        else :
            raise ValueError(f'Model type must be one of \'ridge\', \'lasso\', \'knn\' or \'logistic\', got {model_type}.')

    # Return the average score
    return np.mean(scores, axis=0)

# Grid search for optimal value of k for KNN model

def choose_best_k(X_train, y_train, folds, k_range):
    '''
    Performs a grid search for the optimum value of k using accuracy as the metric to maximise

    INPUTS:
    X_train : a 2D numpy array of training data, each row being one observation
    y_train : a 1D numpy array of the corresponding response variable
    folds : a 2D numpy array, each row consisting of a list of indices of a single fold
    k_range : a 1D numpy array of different values of k to search over

    OUTPUT:
    The optimal choice of k
    '''
    
    k_scores = [] 
    
    for i, k in enumerate(k_range):
        k_scores.append( cross_val_score(X_train, y_train, folds, model_type='knn', k=k) )
        print(f'For {k} neighbours, accuracy = {k_scores[i][0]:.3f}, recall = {k_scores[i][1]:.3f}, precision = {k_scores[i][2]:.3f}, f1 = {k_scores[i][3]:.3f} ')

        # Select best k based on accuracy
        
        best_k_index = np.argmax([score[0] for score in k_scores])
    return k_range[best_k_index]