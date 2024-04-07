import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io


class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        self.img = None
        self.A = io.imread("inputPS1Q3.jpg")
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image
        """
        gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        return gray
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """

        swapImg = np.full(self.A.shape, 0)
        swapImg[:, :, 1] = self.A[:, :, 0] ##R
        swapImg[:, :, 0] = self.A[:, :, 1] ##G
        swapImg[:, :, 2] = self.A[:, :, 2] ##B
        plt.imshow(swapImg)
        plt.show()
        return swapImg

    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """  
        grayImg = self.rgb2gray(self.A)
        plt.imshow(grayImg, cmap=plt.cm.get_cmap("gray"))
        plt.show()
        return grayImg
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        negativeImg = 255 - self.rgb2gray(self.A)
        plt.imshow(negativeImg, cmap=plt.cm.get_cmap("gray"))
        plt.show()
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        mirrorImg = self.rgb2gray(np.fliplr(self.A))
        plt.imshow(mirrorImg, cmap=plt.cm.get_cmap("gray"))
        plt.show()
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        gray = self.rgb2gray(self.A)
        mirror = self.rgb2gray(np.fliplr(self.A))
        avgImg = (gray + mirror) / 2
        avgImg = avgImg.astype('uint8')
        plt.imshow(avgImg, cmap=plt.cm.get_cmap("gray"))
        plt.show()
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            noisyImg, noise: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
            and the noise
        """

        noisyImg = self.rgb2gray(self.A)
        noise = np.random.randint(1,256,noisyImg.shape)
        noisyImg += noise
        noisyImg = np.clip(noisyImg, 0, 255)
       	plt.imshow(noisyImg, cmap = plt.get_cmap('gray'))
        plt.show()
        return noisyImg, noise
        
        
if __name__ == '__main__': 
    
    p3 = Prob3()

    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    noisyImg,_ = p3.prob_3_6()

    




