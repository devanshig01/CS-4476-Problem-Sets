import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""

        self.indoor = io.imread("indoor.png")
        self.outdoor = io.imread("outdoor.png")
       
    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        indoor_red = self.indoor[:, :, 0]
        indoor_green = self.indoor[:, :, 1]
        indoor_blue = self.indoor[:, :, 2]
        outdoor_red = self.outdoor[:, :, 0]
        outdoor_green = self.outdoor[:, :, 1]
        outdoor_blue = self.outdoor[:, :, 2]
        '''
        plt.imshow(indoor_red, cmap='gray')
        plt.show()
        plt.imshow(indoor_green, cmap='gray')
        plt.show()
        plt.imshow(indoor_blue, cmap='gray')
        plt.show()
        plt.imshow(outdoor_red, cmap='gray')
        plt.show()
        plt.imshow(outdoor_green, cmap='gray')
        plt.show()
        plt.imshow(outdoor_blue, cmap='gray')
        plt.show()
        '''
        indoor_lab = cv2.cvtColor(self.indoor, cv2.COLOR_RGB2LAB)
        outdoor_lab = cv2.cvtColor(self.outdoor, cv2.COLOR_RGB2LAB)
        #cv2.imshow('indoor_lab',indoor_lab)
        L, A, B = cv2.split(indoor_lab)
        '''
        cv2.imshow('L', L)
        cv2.imshow('A', A)

        cv2.imshow('B', B)
        '''

        L1, A1, B1 = cv2.split(outdoor_lab)
        '''
        cv2.imshow('L1', L1)
        cv2.imshow('A2', A1)

        cv2.imshow('B3', B1)
        '''

        return

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
       
        np.seterr(invalid='ignore')
        
        RGB = io.imread('inputPS1Q4.jpg') 
        RGB = RGB / 255.0
        RGB = RGB.astype('float64')
        R = RGB[:, :, 0]
        G = RGB[:, :, 1]
        B = RGB[:, :, 2]
        
        V = np.max(RGB, axis = 2)
        m = np.min(RGB, axis = 2)
        
        C = V - m
        S = C
        if (np.all(V == 0)):
            S = 0
        else:
            S = np.divide(C, V)
        Hh = C
        Hh[C == 0] = 0
        Hh[V == R] = (G[V == R] - B[V == R]) / C[V == R]
        Hh[V == G] = ((B[V == G] - R[V == G]) / C[V == G]) + 2
        Hh[V == B] = ((R[V == B] - G[V == B]) / C[V == B]) + 4
        H = Hh / 6
        H[H < 0] += 1

        HSV = np.dstack((H, S, V))  

        ##had to look this up: 
        ##https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

        HSV = np.nan_to_num(HSV)

       
        plt.imshow(HSV, cmap='hsv')
        plt.show()
        return HSV
    
        
if __name__ == '__main__':
    
    p4 = Prob4()
    p4.prob_4_1()
    HSV = p4.prob_4_2()





