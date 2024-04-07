import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super().__init__()

        self.conv_layers = nn.Sequential( 
            nn.Conv2d(1, 10, kernel_size=5, stride=1),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(10, 20, kernel_size=5, stride=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(3))
        self.fc_layers = nn.Sequential(nn.Linear(500, 100), nn.Linear(100, 15))
        self.loss_criterion = nn.CrossEntropyLoss(size_average=None, reduce=None, reduction='sum')
       
        #raise NotImplementedError('SimpleNetDropout not implemented')

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        conv_features = None  # output of x passed through convolution layers (4D tensor)
        flattened_conv_features = None  # conv_features reshaped into 2D tensor using .reshape()
        model_output = None  # output of flattened_conv_features passed through fully connected layers

        conv_features = self.conv_layers(x)
        flattened_conv_features = conv_features.reshape((-1, 500))
        model_output = self.fc_layers(flattened_conv_features)

       
        return model_output
