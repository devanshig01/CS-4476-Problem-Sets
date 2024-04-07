import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    feat = []
    for i in range(X.shape[0]):
        list = []
        list.append(X[i][0] * X[i][1])
        feat.append(list)
    X_new = np.hstack((X, feat))
    return X_new
    


    
