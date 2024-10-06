import torch
import matplotlib.pyplot as plt
import numpy as np

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())

def forward(x):
    return w*x+b 


def criterion(yhat, y):
    return torch.mean((y- yhat)**2)

lr = 0.1
epochs = 7
COST = []
loss = None

for epoch in range(epochs):
    
    # add this to track thhe total loss at each sample
    total = 0

# =============================================================================
    # now iterate all samples here, update after each
# =============================================================================
    for x,y in zip(X, Y):  # key difference between regular and stochastic gradient descent
# =============================================================================


        yhat = forward(x) 
        loss = criterion(yhat, y)  
        loss.backward() 
        w.data = w.data-(lr*w.grad.data) 
        b.data = b.data-(lr*b.grad.data) 
        w.grad.data.zero_()  
        b.grad.data.zero_()
        
        # tracking total loss...
        total += loss.item()
    
    COST.append(total)

print(COST)

plt.plot(COST, marker='o')
plt.title("Cost vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cost (Loss)")
plt.grid(True)