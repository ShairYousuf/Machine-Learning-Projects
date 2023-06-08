# import torch
# from torch import nn
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# #create *known* parameters

# weight = 0.7
# bias = 0.3

# start = 0
# end = 1
# step = 0.02
# X = torch.arange(start, end, step).unsqueeze(dim=1)
# y = weight * X + bias

# #Create a train/test split
# train_split = int(0.8*len(X))
# X_train = X[:train_split]
# Y_train = y[:train_split]
# X_test = X[train_split:]
# y_test = y[train_split:]




# #create linear regression model class

# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weights = nn.Parameter(torch.randn(1,
#                                                 requires_grad=True,
#                                                 dtype=torch.float))
#         self.bias = nn.Parameter(torch.randn(1,
#                                             requires_grad=True,
#                                             dtype=torch.float))
        
#         #forward method to define the computation in the model
#     def forward(self, x:torch.Tensor)->torch.Tensor:
#         return self.weights * x + self.bias
    
# torch.manual_seed(42)
# model_0 = LinearRegressionModel()

# #Make prediction with  model
# with torch.inference_mode():
#     y_preds = model_0(X_test)

# loss_fn = nn.L1Loss()

# optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

# torch.manual_seed(42)

# #Time we loop through data
# epochs = 200

# #Track Different values
# epoch_count = []
# loss_values = []
# test_loss_values = []

# for epoch in range(epochs):
#     model_0.train()

#     y_pred = model_0(X_train)

#     loss = loss_fn(y_pred, Y_train)

#     optimizer.zero_grad()

#     loss.backward()

#     optimizer.step()

#     model_0.eval()

#     with torch.inference_mode():

#         test_pred = model_0(X_test)

#         test_loss = loss_fn(test_pred, y_test)

#     if epoch % 10 == 0:
#         epoch_count.append(epoch)
#         loss_values.append(loss)
#         test_loss_values.append(test_loss)
#         # print(f"Epoch: {epoch}, Training loss: {loss}, Test loss: {test_loss}")
#         # print(model_0.state_dict())

# # with torch.inference_mode():
# #     y_preds_new = model_0(X_test)

# # plot_predictions(predictions=y_preds_new)

# plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label='Training Loss')
# plt.plot(epoch_count, test_loss_values, label='Test Loss')
# plt.title('Training and Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

import torch 
from torch import nn
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weight = 0.7
bias = 0.3
start = 0
end=1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8*len(X))

X_train = X[:train_split]
y_train = y[:train_split]
X_test = X[train_split:]
y_test = y[train_split:]

def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data.squeeze(), train_labels.squeeze(), c='b', label='Training data')
    plt.scatter(test_data.squeeze(), test_labels.squeeze(), c='g', label='Testing data')
    if predictions is not None:
        plt.scatter(test_data.squeeze(), predictions.squeeze(), c='r', label='Predictions')
    plt.legend(prop={'size': 14})
    plt.show()


#plot_predictions(X_train, y_train, X_test, y_test)

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
        
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.linear_layer(x)
    
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
#print(model_1, model_1.state_dict())

model_1.to(device)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

epochs = 200
#Video time 8 hrs 7 mins

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epochs in range(epochs):
    model_1.train()
    y_pred = model_1(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)
    # if epochs % 10 == 0:
    #     print(f"Epoch: {epochs}, Training loss: {loss}, Test loss: {test_loss}")
    #     print(model_1.state_dict())

model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)

# print(y_preds)
# plot_predictions(predictions=y_preds.cpu())

from pathlib import Path

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = 'linear_regression_model_v2.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

loader_model_1 = LinearRegressionModelV2()

loader_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

loader_model_1.to(device)

loader_model_1.eval()

with torch.inference_mode():
    loader_model_1_preds = loader_model_1(X_test)

print(loader_model_1_preds==y_preds)
