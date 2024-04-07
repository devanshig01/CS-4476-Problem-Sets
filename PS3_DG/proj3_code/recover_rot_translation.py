from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def recover_E_from_F(f_matrix: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    '''
    Recover the essential matrix from the fundamental matrix

    Args:
    -   f_matrix: fundamental matrix as a numpy array
    -   k_matrix: the intrinsic matrix shared between the two cameras
    Returns:
    -   e_matrix: the essential matrix as a numpy array (shape=(3,3))
    '''

    e_matrix = None
    #E = K′TFK.
    mat = np.dot(k_matrix.T, f_matrix)
    e_matrix = np.dot(mat, k_matrix)
    return e_matrix
  

def recover_rot_translation_from_E(e_matrix: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    '''
    Decompose the essential matrix to get rotation and translation (upto a scale)

    Ref: Section 9.6.2 

    Args:
    -   e_matrix: the essential matrix as a numpy array
    Returns:
    -   R1: the 3x1 array containing the rotation angles in radians; one of the two possible
    -   R2: the 3x1 array containing the rotation angles in radians; other of the two possible
    -   t: a 3x1 translation matrix with unit norm and +ve x-coordinate; if x-coordinate is zero then y should be positive, and so on.

    '''

    R1 = None
    R2 = None
    t = None
    
    #taken from the article
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    U, diag, V = np.linalg.svd(e_matrix)
    #V = V^T

    R1 = np.dot((np.dot(U, W)), V)
    R2 = np.dot(np.dot(U, W.T), V)

    R1 = Rotation.from_matrix(R1).as_rotvec()
    R2 = Rotation.from_matrix(R2).as_rotvec()

    #print(U)
    #last column of U
    t = U[:, 2]
   
    #print(t)

    if t[1] < 0:
        t *= -1
    return R1, R2, t

  
    
