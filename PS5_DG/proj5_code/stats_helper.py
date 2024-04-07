import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
    """
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    """
    scaler = StandardScaler()

    
    for image in os.listdir(dir_name):
        image = os.path.join(dir_name, image)
        #Checking if it is a valid directory
        if os.path.isdir(image):

            for path in os.listdir(image):
                #Categories for data folder (i.e forest)
                path = os.path.join(image, path)
            
                for path2 in os.listdir(path):
                    #Accessing the actual images in each folder now.

                    #print(path2)
                    path2 = os.path.join(path, path2)
                    
                    image_new = Image.open(path2).convert('L')
                    image_new = Image.Image.split(image_new)
                    image_new = np.array(image_new[0])

                    image_new = np.reshape(image_new, (-1, 1))
                    scaler.partial_fit(image_new)
    
    mean = (scaler.mean_) / 255
    std = (np.sqrt(scaler.var_)) / 255


    return mean, std

  