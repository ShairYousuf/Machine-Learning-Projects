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

print(X[:10],y[:10])