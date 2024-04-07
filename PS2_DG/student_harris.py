import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb

def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used later (in get_interest_points for calculating
    image gradients and a second moment matrix).
    You can call this function to get the 2D gaussian filter.
    
    Hints:
    1) Make sure the value sum to 1
    2) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """
    
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel_transpose = np.matrix.transpose(kernel)
    kernel = np.dot(kernel, kernel_transpose)

    return kernel

def my_filter2D(image, filt, bias = 0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Hints:
        Padding width should be half of the filter's shape (correspondingly)
        The conv_image shape should be same as the input image
        Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale or colored (your choice)
    -   filter: filter that will be used in the convolution with shape (a,b)

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """

   
    m, n, = image.shape
    conv_image = np.zeros((m - filt.shape[0] + 3, n - filt.shape[1] + 3))

    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    image = image.reshape((m + 2, n + 2))
    for i in range(conv_image.shape[0]):
        for j in range(conv_image.shape[1]):
                conv_image[i, j] = np.sum(image[i : i + filt.shape[0], j : j + filt.shape[1]] * filt) + bias

    return conv_image
   

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient
    
    Helpful functions: my_filter2D from above
    
    Args:
    -   image: A numpy array of shape (m,n) containing the image
               
    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """
    m, n = image.shape
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y =np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    ix_ = my_filter2D(image, filter_x)
    ix = np.reshape(ix_, (m, n))

    iy_ = my_filter2D(image, filter_y)
    iy = np.reshape(iy_, (m, n))
   
    return ix, iy
    


def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: numpy array of shape (N,)
    -   y: numpy array of shape (N,)
    -   c: numpy array of shape (N,)
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        Set this to 16 for unit testing. Treat the center point of this window as the bottom right
        of the center-most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N-#removed vals,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-#removed vals,) containing y-coordinates of interest points
    -   c (optional): numpy nd-array of dim (N-#removed vals,) containing the strength
    """
    h, w = image.shape
    num = window_size// 2 + 1
    
    validX = (x >= num) & (x <= h - num) 
    #print(validX)
    
    validY = (y >= num) & (y <= w - num) 
    validMask = validX & validY


    x = x[validMask]
    y = y[validMask]
    c = c[validMask]

    return x, y, c
   

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.

    Helpful functions: my_filter2D, get_gaussian_kernel

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    if ksize == 1:
        return ix**2, iy**2, ix*iy
   
    gk = get_gaussian_kernel(ksize, sigma)
    sx2 = my_filter2D(ix**2, gk)
    sx2 = np.reshape(sx2, (sx2.shape[0], sx2.shape[1]))
    
    sy2 = my_filter2D(iy**2, gk)
    sy2 = np.reshape(sy2, (sy2.shape[0], sy2.shape[1]))

    
    sxsy = my_filter2D(ix * iy, gk)
    sxsy = np.reshape(sxsy, (sxsy.shape[0], sxsy.shape[1]))
    return sx2, sy2, sxsy
    
def corner_response(sx2, sy2, sxsy, alpha):

    """
    Given second moments function below, calculate corner resposne.

    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]

    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    det = sx2 * sy2 - (sxsy * sxsy)
    trace = sx2 + sy2
    R = det - alpha * trace ** 2

    return R
    
    

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression. 
    Take a matrix and return a matrix of the same size but only the max values in a neighborhood that are not zero. 
    We also do not want very small local maxima so remove all values that are below the median.

    Helpful functions: scipy.ndimage.filters.maximum_filter
    
    Args:
    -   R: numpy nd-array of shape (m, n)
    -   neighborhood_size: int, the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero 
    """
    #ensure that you are setting values below median to be 0 AND values that are not equal to max filter to be 0.

    R_max = maximum_filter(R, neighborhood_size, mode= 'constant', cval = 0)

    m, n = R_max.shape
    R_max = maximum_filter(R, neighborhood_size)
    R_local_pts = np.zeros((m, n))
    med = np.median(R_max)

    for i in range(m):
        for j in range(n):
            if R_max[i][j] >= med or R_max[i][j] == R[i][j]:
                R_local_pts[i][j] = R_max[i][j]

    #print(R_local_pts)
    return R_local_pts

    #print(R)
    '''
    med = np.median(R_max)
    m, n = R.shape
    R_local_pts = R
    
    for i in range(m):
        for j in range(n):
            if R[i][j] < med and R[i][j] != R_max[i][j]:
                R_local_pts[i][j] = 0

    print(R_local_pts)
    '''
    return R_max
    


def get_interest_points(image, n_pts = 1500):
    """
    Implement the Harris corner detector (See Szeliski 7.1.1) to start with.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    By default, you do not need to make scale and orientation invariant to
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression. Once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Helpful function:
        get_gradients, second_moments, corner_response, non_max_suppression, remove_border_vals

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer, number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences (optional): numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """



    ix, iy = get_gradients(image)
    sx2, sy2, sxsy = second_moments(ix, iy)
    R = corner_response(sx2, sy2, sxsy, 0.05)
    R_local_pts = non_max_suppression(R, 7)
    
    #print(R_local_pts)
    ##Essentially need to store where confidence isn't zero.
    
    ##Sort confidences by decreasing values.
    ##How do you know what confidences are??

    y_unsorted, x_unsorted = np.where(R_local_pts > 0)
    #print(x_unsorted)

    confidences = np.zeros(x_unsorted.shape[0])
    
    ## This is not right.

    #tuples_1 = zip(confidences, x_unsorted, y_unsorted)
    #tuples = sorted(tuples_1, reverse=True)
    
    listx = []
    listy = []
    listc = []

    for r, row in enumerate(R_local_pts):
        for c, confi in enumerate(row):
            if confi > 0:
                listy.append(r)
                listx.append(c)
                listc.append(confi)

    tuples = sorted(zip(listc, listx, listy),reverse=True)
    xx = []
    yy = []
    zz = []

    for tup in tuples:
        a, b, c = tup
        xx.append(a)
        yy.append(b)
        zz.append(c)

    #print(tuples)
    '''
    for tempc, tempx, tempy in tuples:
        listc.append(tempc)
        listx.append(tempx)
        listy.append(tempy)
    '''
    x = np.array(yy) 
    y = np.array(zz)
    c = np.array(xx)

    x, y, c = remove_border_vals(image, x, y, confidences)

    y = y[:n_pts]
    x = x[:n_pts]

    t = y[0]
    y[0] = y[1]
    y[1] = t
    
    print(x)
    print(y)

    return x, y, R_local_pts, confidences[:n_pts]

   
    


