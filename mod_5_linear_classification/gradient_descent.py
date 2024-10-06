import torch.nn as nn
from toy_set import toy_set
from logistic_regression import logistic_regression
from torch.utils.data import DataLoader
from torch import optim
import sys


class logistic_reg(nn.Module):
    def __init__(self, in_dim):
        super(logistic_reg, self).__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        out = nn.Sigmoid()(self.linear(x))
        return out


criterion = nn.BCELoss()

ds = toy_set()

tl = DataLoader(dataset = ds, batch_size=1)

model = logistic_regression(1)

optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    
    cost = 0
    
    for x,y in tl:
        
        yhat = model(x)  # make predictions (forward is implicitly called)
        loss = criterion(yhat, y)  # evaluate predictions
        
        optimizer.zero_grad()  # zero gradient before calculating
        
        loss.backward()  # calculate partial derivatives 
        optimizer.step()  # update parameters
        cost += loss
    
    print(cost.item())
        