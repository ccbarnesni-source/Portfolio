import numpy as np

def logistic(x):
    '''
    The logistic function

    INPUTS:
    x - a numpy array

    OUTPUTS;
    The logistic function applied element wise to x
    '''
    
    return 1. / (1. + np.exp(-x))

def predict_log(X, beta, beta_0):
    '''
    Obtains the predicted probabilities, p_i, given training data and parameters beta

    INPUTS:
    X - a n x p array of training data
    beta - a p x 1 array of parameters for beta
    beta_0 - float, the intercept parameter

    OUTPUTS:
    p - an n-vector of predicted probabilities
    '''
    
    p = logistic(X @ beta + beta_0)
    return p.squeeze() # Convert to a 1D array

def logistic_grads(X, y, beta, beta_0):
    '''
    Computes the cost function (negative log likelihood) and gradients for given parameters

    INPUTS:
    X - an n x p array of observations
    y - an n x 1 array of responses
    beta - a p x 1 array of logistic model coefficients
    beta_0 - float, the intercept term

    OUPUTS:
    grads - a dictionary with the components of the derivatives
    cost - the negative log likelihood
    '''

    # Obtain the initial probabilities
    p = predict_log(X, beta, beta_0)

    # Using gradient descent so we must minimise the negative of the log likelihood function
    cost = - (y * np.log(p) + (1-y) * np.log(1 - p)).mean()

    # Calculate derivatives
    dbeta = 0

    assert y.shape == p.shape, f'y and p do not have the same shape. y has shape {y.shape} and p has shape {p.shape}'
    
    for i in range(len(y)):
        dbeta += X[i,:] * (p[i] - y[i])
    
    dbeta = (dbeta/X.shape[0]).reshape(-1,1)
    dbeta_0 =  (p - y).mean()

    assert(dbeta.shape==beta.shape)
    assert(dbeta_0.dtype==float)
  
    # Store gradients in a dictionary
    grads = {"dbeta": dbeta, "dbeta_0": dbeta_0}
  
    return grads, cost

def optimise(X, y, num_iterations=5000, learning_rate=0.005, print_cost=False):
    '''
    Implements gradient descent to minimise the negative log likelihood function for logistic regression

    INPUTS:
    X - a n x p array of observations
    y - a n x 1 array of corresponding responses
    beta - a p x 1 array of the initial model parameters (not including intercept term)
    beta_0 - float, the initial value of the intercept term
    num_iterations - how many iterations to use in gradient descent, default is 5000
    learning_rate - the learning rate for gradient descent
    print_cost - default False, if set to True will print the cost function every 100 iterations

    OUTPUTS:
    params - the final parameters after the gradient descent algorithm has finished
    grads - the final gradients after gradient descent has been implemented
    costs - a list of the cost function every 100 iterations, useful for debugging purposes
    '''

    # Initialise the vector beta and the corresponding intercept term
    beta = np.zeros(shape = (X.shape[1],1), dtype=float)
    beta_0 = 0
    
    costs = []
    for i in range(num_iterations):

        # Calculate cost and gradients
        grads, cost = logistic_grads(X, y, beta, beta_0)
      
        # retrieve derivatives from grads
        dbeta = grads["dbeta"]
        dbeta_0 = grads["dbeta_0"]
      
        # updating procedure
        beta = beta - learning_rate * dbeta
        beta_0 = beta_0 - learning_rate * dbeta_0
      
        # record the costs
        if i % 100 == 0:
            costs.append(cost)
      
        # print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print ("cost after iteration %i: %f" %(i, cost))
  
    # save parameters and gradients in dictionary
    params = {"beta": beta, "beta_0": beta_0}
    grads = {"dbeta": dbeta, "dbeta_0": dbeta_0}
  
    return params, grads, costs

def predict(X_test, beta, beta_0, thres=0.5):
    '''
    Classifies a set of observations using a logistic model
    
    INPUTS:
    X_test - an n x p array of observations
    beta - a p x 1 array of logisitic model parameters
    beta_0 - float, the intercept term
    thres - the threshold hyperparameter at which a given observation is classified as belonging to the target class

    OUTPUTS:
    an n x 1 array of predicted classifications
    '''

    # Check threshold is an appropriate value
    if ((thres<0) | (thres>1)):
        raise ValueError(f'Threshold value must be in the interval [0,1], got {thres}.')
    
    n = X_test.shape[0]
  
    # Compute vector of probabilities
    p = predict_log(X_test, beta, beta_0)

    return np.astype((p>=thres), int)