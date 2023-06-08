import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
n_samples = 1000

X,y = make_circles(n_samples, noise=0.03, random_state=42)


circles = pd.DataFrame({"X0":X[:,0],
                        "X1":X[:,1], 
                        "label":y})

# plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()
#Sklearn has some great datasets for projects

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=8)
        self.layer_2 = nn.Linear(in_features=8, out_features=1)
        
    def forward(self, x):
        return self.layer_2(self.layer_1(x))

