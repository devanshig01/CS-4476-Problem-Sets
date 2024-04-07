import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

'''
We are going to use the California housing prices dataset provided by sklearn
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
to train a 2 fully connected layer neural net. We are going to buld the neural network from scratch.
'''


class dlnet:

    def __init__(self, x, y, lr = 0.01):
        '''
        This method initializes the class, it is implemented for you. 
        Args:
            x: data
            y: labels
            Yh: predicted labels
            dims: dimensions of different layers
            param: dictionary of different layers parameters
            ch: Cache dictionary to store forward parameters that are used in backpropagation
            loss: list to store loss values
            lr: learning rate
            sam: number of training samples we have

        '''        
        self.X=x # features
        self.Y=y # ground truth labels

        self.Yh=np.zeros((1,self.Y.shape[1])) # estimated labels
        self.dims = [8, 15, 1] # dimensions of different layers

        self.param = { } # dictionary for different layer variables
        self.ch = {} # cache for holding variables during forward propagation to use them in back prop
        self.loss = [] # list to store loss values

        self.iter = 0 # iterator to index into data for making a batch 
        self.batch_size = 64 # batch size 
        
        self.lr=lr # learning rate
        self.sam = self.Y.shape[1] # number of training samples we have
        self._estimator_type = 'classifier'
        self.neural_net_type = "Relu -> Tanh" 


    def nInit(self): 
        '''
        This method initializes the neural network variables, it is already implemented for you. 
        Check it and relate to the mathematical description above.
        You are going to use these variables in forward and backward propagation.
        '''   
        np.random.seed(1)
        self.param['theta1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['theta2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))     


    def Relu(self, u):
        '''
        In this method you are going to implement element wise Relu. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension. 
        Input: u of any dimension
        return: Relu(u) 

        Recall Hint 1 from the notebook
        '''
        u_new = np.copy(u)
        u_new[u_new < 0] = 0
        return u_new

        
        

    def Tanh(self, u):
        '''
        In this method you are going to implement element wise Tanh. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension. 
        Input: u of any dimension
        return: Tanh(u) 

        Recall Hint 1 from the notebook
        '''
        u_new = np.copy(u)
        tanh = (np.exp(u_new) - np.exp(-u_new)) / (np.exp(u_new) + np.exp(-u_new))
        return tanh
        
    
    
    def dRelu(self, u):
        '''
        This method implements element wise differentiation of Relu, it is already implemented for you.  
        Input: u of any dimension
        return: dRelu(u) 
        '''
        u[u<=0] = 0
        u[u>0] = 1
        return u


    def dTanh(self, u):
        '''
        This method implements element wise differentiation of Tanh, it is already implemented for you.
        Input: u of any dimension
        return: dTanh(u) 
        '''
        
        o = np.tanh(u)
        return 1-o**2
    
    

    def nloss(self,y, yh):
        '''
        In this method you are going to implement mean squared loss. 
        Refer to the description above and implement the appropriate mathematical equation.
        Input: y 1xN: ground truth labels
               yh 1xN: neural network output

        return: MSE 1x1: loss value 
        '''
        num = (1.0 / (2 * y.shape[1]))
        MSE = np.sum(np.square((y - yh))) * num
        return MSE


    def forward(self, x):
        '''
        Fill in the missing code lines, please refer to the description for more details.
        Check nInit method and use variables from there as well as other implemented methods.
        Refer to the description above and implement the appropriate mathematical equations.
        do not change the lines followed by #keep. 

        Input: x DxN: input
        return: o2 1xN
        '''  
            
        self.ch['X'] = x #keep

        u1 = np.dot(self.param['theta1'], x) + self.param['b1']
        o1 = self.Relu(u1)
        u2 = np.dot(self.param['theta2'], o1) + self.param['b2']
        o2 = self.Tanh(u2)

        self.ch['u1'],self.ch['o1']=u1,o1 #keep 
        self.ch['u2'],self.ch['o2']=u2,o2 #keep

        return o2 #keep
    

    def backward(self, y, yh):
        '''
        Fill in the missing code lines, please refer to the description for more details
        You will need to use cache variables, some of the implemented methods, and other variables as well
        Refer to the description above and implement the appropriate mathematical equations.
        do not change the lines followed by #keep.  

        Input: y 1xN: ground truth labels
               yh 1xN: neural network output

        Return: dLoss_theta2 (1x15), dLoss_b2 (1x1), dLoss_theta1 (15xD), dLoss_b1 (15x1)

        Recall Hint 2 from the notebook
        '''    
        #TODO: implement this 
             
        dLoss_o2 = np.subtract(self.ch['o2'], y)
        dLoss_o2 /= y.shape[1]

        dLoss_u2 = np.multiply(dLoss_o2, self.dTanh(self.ch['u2']))

        dLoss_theta2 = np.matmul(dLoss_u2, self.ch['o1'].T)

        dLoss_b2 = np.matmul(dLoss_u2, np.ones(dLoss_u2.shape).T)

        dLoss_o1 = np.dot(self.param["theta2"].T, dLoss_u2) 
        dLoss_u1 = np.multiply(dLoss_o1, self.dRelu(self.ch['u1']))
        dLoss_theta1 = np.matmul(dLoss_u1, self.ch['X'].T)
        dLoss_b1 = np.matmul(dLoss_u1, np.ones(dLoss_u2.shape).T)  


        # parameters update, no need to change these lines
        self.param["theta2"] = self.param["theta2"] - self.lr * dLoss_theta2 #keep
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2 #keep
        self.param["theta1"] = self.param["theta1"] - self.lr * dLoss_theta1 #keep
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1 #keep

        return dLoss_theta2, dLoss_b2, dLoss_theta1, dLoss_b1

    def gradient_descent(self, x, y, iter = 60000):
        '''
        This function is an implementation of the gradient descent algorithm.
        Note:
        1. GD considers all examples in the dataset in one go and learns a gradient from them. 
        2. One iteration here is one round of forward and backward propagation on the complete dataset. 
        3. Append loss at multiples of 1000 i.e. at 0th, 1000th, 2000th .... iterations to self.loss

        Input: x DxN: input
               y 1xN: labels
        ''' 
        self.nInit()
        for i in range(iter):
            yh = self.forward(x)
            l = self.nloss(y, yh)
            self.loss.append(l)
            self.backward(y, yh)

            if i % 1000 == 0:
                print('Loss after iteration ' + str(i) + ': ' + str(l))
            

    #bonus for undergrdauate students 
    def batch_gradient_descent(self, x, y, iter = 60000):
        '''
        This function is an implementation of the batch gradient descent algorithm

        Note: 
        1. Batch GD loops over all mini batches in the dataset one by one and learns a gradient 
        2. One iteration here is one round of forward and backward propagation on one minibatch. 
           You will use self.iter and self.batch_size to index into x and y to get a batch. This batch will be
           fed into the forward and backward functions.
        3. Append loss at multiples of 1000 iterations i.e. at 0th, 1000th, 2000th .... iterations  
        4. It is fine if you get a noisy plot since learning on a batch adds variance to the 
           gradients learnt
        5. Be sure that your batch size remains constant (***see notebook for more detail***).

        Input: x DxN: input
               y 1xN: labels
        '''
        self.nInit()
        start = 0

        for i in range(iter):
            index_list = [k % y.shape[1] for k in range(start, start + self.batch_size)]
            x_new = x[:, index_list]
            y_new = y[:, index_list]
            yh = self.forward(x_new)
            l = self.nloss(y_new, yh)

            if i % 1000 == 0:
                self.loss.append(l)
                print('Loss after iteration ' + str(i) + ': ' + str(l))
            
            self.backward(y_new, yh)
            start = start + self.batch_size


    def predict(self, x): 
        '''
        This function predicts new data points
        Its implemented for you

        Input: x DxN: inputs
        Return: y 1xN: predictions

        '''
        Yh = self.forward(x)
        return Yh
