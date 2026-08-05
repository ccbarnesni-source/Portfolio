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

    # If pandas series/dataframe is given, convert to a numpy array first
    y = np.asarray(y)
    
    y_bar = np.mean(y)
    SST = np.sum((y-y_bar)**2)    # Sum of squares total, i.e. the variance of the target variable
    SSE = np.sum((y-y_pred)**2)   # Sum of squared errors
    R = 1 - SSE/SST
    return R

def MSE(y, y_pred):
    '''
    Computes the MSE (mean square error) of a given array.

    INPUTS:
    y - the observed response
    y_pred - the predicted response from the model

    OUTPUTS:
    MSE - the mean squared error of the prediction
    '''

    # If pandas series/dataframe is given, convert to a numpy array first
    y = np.asarray(y)
    
    MSE = np.sum((y-y_pred)**2)/len(y)
    return MSE

# Define a score function for classifiers

def classification_score(y, y_pred):
    '''
    Returns the accuracy, recall, precision and f1-score of the KNN model given training and test sets

    INPUTS:
    y_train : a 1D numpy array of the corresponding response variable
    y_test : a 1D numpy array of the corresponing response variable to the test data


    OUTPUTS:
    acc - the accuracy of the model
    recall - the recall of the predictions
    precision - the precision of the predictions
    f1_score - the harmonic mean of recall and precision
    '''

    # If pandas series/dataframe is given, convert to a numpy array first
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    TP = np.sum( (y_pred == y) & (y==1) )
    TN = np.sum( (y_pred == y) & (y==0) )
    FP = np.sum( (y_pred != y) & (y==0) )
    FN = np.sum( (y_pred != y) & (y==1) )

    acc = np.mean(y_pred==y)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = ( 2*recall*precision ) / ( recall + precision ) 
    
    return [acc, recall, precision, f1_score]