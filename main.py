import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#create *known* parameters

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

#Create a train/test split
train_split = int(0.8*len(X))
X_train = X[:train_split]
Y_train = y[:train_split]
X_test = X[train_split:]
y_test = y[train_split:]

def plot_predictions(train_data=X_train, 
                     train_labels=Y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', label='Training data')
    plt.scatter(test_data, test_labels, c='g', label='Testing data')
    if predictions is not None:
        plt.scatter(train_data, predictions, c='r', label='Predictions')
    plt.legend(prop = {'size': 14})
    plt.show()

plot_predictions()

#create linear regression model class

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
        
        #forward method to define the computation in the model
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.weights * x + self.bias
