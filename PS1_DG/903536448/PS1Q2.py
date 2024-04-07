import numpy as np
import matplotlib.pyplot as plt

class Prob2():
    def __init__(self):
        """Load inputAPS1Q2.npy here as a class variable A."""
        self.A = np.load("inputAPS1Q2.npy")
        pass
        
    def prob_2_1(self):
        """Do plotting of intensities of A in decreasing value."""
        x = np.array(self.A).flatten()
        decreasing = np.sort(x)[::-1]
        plt.imshow([decreasing], cmap='gray', aspect='auto')
        plt.show()
        return 
    
    def prob_2_2(self):
        """Display histogram of A's intensities with 20 bins here."""
        x = np.array(self.A).flatten()
        plt.hist(x, bins=20)
        plt.show()
        return
    
    def prob_2_3(self):
        """
        Create a new matrix X that consists of the bottom left quadrant of A here.
        Returns:
            X: bottom left quadrant of A which is of size 50 x 50
        """ 
        X = self.A[50: , :50]
        return X
    
    def prob_2_4(self):
        """Create a new matrix Y, which is the same as A, but with A’s mean intensity value subtracted from each pixel.
        Returns:
            Y: A with A's mean intensity subtracted from each pixel. Output Y is of size 100 x 100.
        """
        mean = np.mean(self.A[::-1])
        Y = self.A - mean
        return Y
    
    def prob_2_5(self):
        """
        Create your threshholded A i.e Z here.
        Returns:
            Z: A with only red pixels when the original value of the pixel is above the threshhold. Output Z is of size 100 x 100.
        """
        mean = np.mean(self.A[::-1])
        Z = np.full((100,100,3), 0)
        Z[self.A > mean] = np.array([1, 0 ,0])
        return Z


if __name__ == '__main__':
    
    p2 = Prob2()
    p2.prob_2_1()
    p2.prob_2_2()
    X = p2.prob_2_3()
    Y = p2.prob_2_4()
    Z = p2.prob_2_5()