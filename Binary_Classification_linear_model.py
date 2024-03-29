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


plot_predictions(X_train, y_train, X_test, y_test)

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
        
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.linear_layer(x)
    
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()

model_1.to(device)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

epochs = 200

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
model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)


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
