from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU


class CNN(object):
    def __init__(self):
        # change these to appropriate values
        self.batch_size = 128
        self.epochs = 10
        self.init_lr= 1e-3 #learning rate

        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''

        self.model = Sequential()
        self.model.add(Conv2D(filters=8, kernel_size =(3, 3), strides=(1, 1), padding='same',
                               input_shape=(28,28,1)))
        self.model.add(LeakyReLU(alpha = '-0.1'))
        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
        self.model.add(LeakyReLU(alpha = '-0.1'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.30))
        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
        self.model.add(LeakyReLU(alpha = '-0.1'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        self.model.add(LeakyReLU(alpha = '-0.1'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.30))
        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(LeakyReLU(alpha = '-0.1'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        #There is no leakyrelu?
        self.model.add(Activation('relu'))
     

      
        return self.model
        
        

    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use model.compile() to build your model.
        '''
        self.model = model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        

        return self.model
