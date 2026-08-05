import numpy as np
import tools.cross_validation

# Function to measure Euclidean distance

def euclidean_distance(p, q):
    '''
    Takes 2D arrays and returns the euclidean distance between each vector in the array

    Inputs:
    p : a 2D numpy array
    q : a 2D numpy array

    Output:
    The euclidean distance for every pair of rows in x and y
    '''
    return np.sqrt(((q-p)**2).sum(axis=1))

# Find the k-nearest neighbours

def k_neighbours(X_train, X_val, k=5, return_distance=False):
    '''
    Finds the k-nearest neighbours to each entry in a validation set

    INPUTS:
    X_train : The data on which the model is trained. The nearest neighbours will be selected from this set
    X_val : The set of new inputs, each of which we are trying to find the k-nearest neighbours
    k : The number of neighbours to select
    return_distance : Boolean, gives the option to return the distance of each neighbour as well

    OUTPUTS:
    neigh_ind : a 2D numpy array containing the indices of each of the nearest neighbours
    '''
    
    dist = []
    neigh_ind = []
  
    # Compute distance from each point x in the validation set to all points in the training set
    
    N = X_val.shape[0]
    point_dist = [euclidean_distance(X_val[n,:], X_train) for n in range(N) ]

    # Determine which k training points are closest to each validation point
    
    for row in point_dist:
    
        enum_neigh = enumerate(row)
        sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:k]

        ind_list = [tup[0] for tup in sorted_neigh]
        dist_list = [tup[1] for tup in sorted_neigh]

        dist.append(dist_list)
        neigh_ind.append(ind_list)

        # Return distances together with indices of k nearest neighbours
        
        if return_distance:
            return np.array(dist), np.array(neigh_ind)
    return np.array(neigh_ind)

# Define a function for classifying the data

def predict(X_train, y_train, X_test, k=5):
    '''
    Given training data, returns the predicted class on a set of new inputs

    INPUTS:
    X_train : a 2D numpy array, each row consisting of a single observation of the training data for the classifier
    y_train : a 1D numpy array corresponding to the target classification
    X_test : a 2D numpy array, each row consisting of a single observation of the new data to classify
    k : the number of neighbours to use, default is 5

    OUTPUT:
    y_pred : a 1D numpy array of the predicted classes on the test set
    '''

    # Check hyperparameter is valid
    if not isinstance(k, int):
        raise TypeError(f'k must be an integer, got {type(k)}.')
    
    # Obtain the k-nearest neighbours
    neighbours = k_neighbours(X_train, X_test, k=k)
    
    # Count number of occurences of label with np.bincount and choose the label that has most with np.argmax
    M = len(neighbours)
    y_pred = np.array([ np.argmax(np.bincount(np.array(y_train[neighbours[m]], dtype=int))) for m in range(M)])
    
    return y_pred

# Grid search for optimal value of k

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