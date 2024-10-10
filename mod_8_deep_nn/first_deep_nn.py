import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
from Data import Data
from Net import Net
torch.manual_seed(1)

# =============================================================================
#
# =============================================================================

def plot_decision_regions_3class(model, data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1 
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _, yhat = torch.max(model(XX), 1)
    yhat = yhat.numpy().reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:] == 0, 0], X[y[:] == 0, 1], 'ro', label = 'y=0')
    plt.plot(X[y[:] == 1, 0], X[y[:] == 1, 1], 'go', label = 'y=1')
    plt.plot(X[y[:] == 2, 0], X[y[:] == 2, 1], 'o', label = 'y=2')
    plt.title("decision region")
    plt.legend()
    
# =============================================================================
#
# =============================================================================

def train(data_set, model, criterion, train_loader, optimizer, epochs=100):
    LOSS = []
    ACC = []
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
        ACC.append(accuracy(model, data_set))
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(LOSS, color = color)
    ax1.set_xlabel('Iteration', color = color)
    ax1.set_ylabel('total loss', color = color)
    ax1.tick_params(axis = 'y', color = color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color = color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color = color)
    ax2.tick_params(axis = 'y', color = color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.show()
    return LOSS
# =============================================================================
#
# =============================================================================
def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()
# =============================================================================
# load data
# =============================================================================
data_set = Data()
data_set.plot_stuff()
data_set.y = data_set.y.view(-1)
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#
#
#
# =============================================================================
# we will now train 3 nn
# =============================================================================
#
# =============================================================================
# 1. Train the model with 1 hidden layer with 50 neurons
# =============================================================================
# Layers = [2, 50, 3]
# model = Net(Layers)
# learning_rate = 0.10
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# train_loader = DataLoader(dataset=data_set, batch_size=20)
# criterion = nn.CrossEntropyLoss()
# LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs=100)

# plot_decision_regions_3class(model, data_set)

# print(Net([3,3,4,3]).parameters)
# =============================================================================
# 2. Train the model with 2 hidden layers with 20 neurons
# =============================================================================
# Layers = [2, 10, 10, 3]
# model = Net(Layers)
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# train_loader = DataLoader(dataset=data_set, batch_size=20)
# criterion = nn.CrossEntropyLoss()
# LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs=1000)

# plot_decision_regions_3class(model, data_set)

# =============================================================================
# 3. Create a network with three hidden layers each with ten neurons.
# =============================================================================
Layers = [2, 10, 10, 10, 3]
model = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
train_loader = DataLoader(dataset = data_set, batch_size = 20)
criterion = nn.CrossEntropyLoss()
LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs = 1000)
plot_decision_regions_3class(model, data_set)




























