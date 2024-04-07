import time
from typing import Tuple

import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares


def objective_func(x: np.ndarray, **kwargs):
    """
    Calculates the difference in image (pixel coordinates) and returns 
    it as a 2*n_points vector

    Args: 
    -        x: numpy array of 11 parameters of P in vector form 
                (remember you will have to fix P_34=1) to estimate the reprojection error
    - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                retrieve these 2D and 3D points and then use them to compute 
                the reprojection error. 
                To get the 2D points, use kwargs['pts2d']
                To get the 3D points, use kwargs['pts3d']
    Returns:
    -     diff: A 2*N_points-d vector (1-D numpy array) of differences betwen 
                projected and actual 2D points

    """

    diff = None
    #Add the one at the end so we can form a 3*4 matrix easily.
    x = np.append(x, 1.0)

    ##Form matrix P with dimensions 3 * 4.
    P = np.ones((3, 4))
    P[0] = x[0:4]
    P[1] = x[4:8]
    P[2] = x[8: 12]

    
    points_2d = kwargs['pts2d']
    points_3d = kwargs['pts3d']

    points = projection(P, points_3d)
    diff = points - points_2d
    list = []

    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            list.append(diff[i][j])
    
    diff = np.array(list)
    return diff
    


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in non-homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """

    ## Convert points_3d into homogenous coordinates. --> add ones to the matrix.
    
    points_3d_new = np.append(points_3d, np.ones((points_3d.shape[0], 1)), axis = 1)

    #Dot product with the projection matrix.
    arr = np.dot(P, points_3d_new.T)

    #finding coordinates for x_i and y_i
    arr[0] = arr[0] / arr[2]
    arr[1] = arr[1] / arr[2]

    new_arr = arr[0: 2]

    #Transform the matrix because [x_{i}, ..]
                                # [y_{i}, ..] --> [x_{i}, y_{i}]

    projected_points_2d = new_arr.T
    
    return projected_points_2d

    

################# UPDATE function name to estimate_projection_matrix here or update in notebook #########################
def estimate_camera_matrix(pts2d: np.ndarray,
                           pts3d: np.ndarray,
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squares form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1) 
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix 

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters. 

              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.

              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables 
                          for the objective function
    '''

    

    start_time = time.time()
    
    P = least_squares(objective_func, initial_guess.flatten()[0:11], method='lm', verbose=2, max_nfev=50000, 
        kwargs = {'pts2d': pts2d,
              'pts3d': pts3d})


    print("Time since optimization start", time.time() - start_time)
    P = np.append(P.x, 1.0)
   
    P = np.reshape(P, (3, 4))
    return P


def decompose_camera_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix

        Args:
        -  P: 3x4 numpy array projection matrix

        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''
    val = rq(P[:, 0:3])
    K, R = val
   
    return K, R


def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray,
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    cc = None
    cc = np.dot(np.linalg.inv(np.dot(K, R_T)), P[:, 3]).flatten()
    cc = cc * -1
    return cc

 