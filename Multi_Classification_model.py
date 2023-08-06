import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
NUM_CLASSES=4
NUM_FEATURES=2
RANDOM_SEED=42
from helper_functions import plot_decision_boundary, plot_predictions

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

X_blob, y_blob = make_blobs(n_samples=1000, 
                            n_features=NUM_FEATURES, 
                            centers=NUM_CLASSES, 
                            cluster_std=1.5, 
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, 
                                                                        test_size=0.2, 
                                                                        random_state=RANDOM_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

modelV0 = BlobModel(input_features=2, output_features=4, hidden_units=8).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=modelV0.parameters(), lr=0.1)

modelV0.eval()
with torch.inference_mode():
    y_pred = modelV0(X_blob_test.to(device))

y_pred_probs = torch.softmax(y_pred, dim=1)
        
y_preds = torch.argmax(y_pred_probs, dim=1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

X_blob_train = X_blob_train.to(device)
y_blob_train = y_blob_train.to(device)
X_blob_test = X_blob_test.to(device)
y_blob_test = y_blob_test.to(device)

for epochs in range(epochs):
    modelV0.train()
    y_logits = modelV0(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_blob_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    modelV0.eval()
    with torch.inference_mode():
        test_logits = modelV0(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_acc = accuracy_fn(y_blob_test, test_pred)
        test_loss = loss_fn(test_logits, y_blob_test)

    if epochs % 10 == 0:
        print(f"Epoch: {epochs} | Train Loss: {loss:.2f} | Train Acc: {acc:.2f}% | Test Loss: {test_loss:.2f} | Test Acc: {test_acc:.2f}%")

modelV0.eval()
with torch.inference_mode():
    y_logits = modelV0(X_blob_test.to(device))

y_pred_probs = torch.softmax(y_logits, dim=1)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(modelV0, X_blob_train, y_blob_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(modelV0, X_blob_test, y_blob_test)
plt.show() 
