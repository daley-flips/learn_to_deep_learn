import torch
import torch.nn as nn
from torch import sigmoid
import sys
# =============================================================================
class Net(nn.Module):
# =============================================================================
    # D_in: dimension of input (x)
    
    # H is how many neurons we have 
    # layer size
        # it is also the number of inputs for the second layer
    
    # D_out is the dimension of output (y)

    
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)  # linear function for first hidden layer (superscipt 1)
        self.linear2 = nn.Linear(H, D_out)  # linear function for output layer (superscipt 2)
# =============================================================================
#
# =============================================================================
    def forward(self, x):
        
        # note that linear1 runs the linear function for  all neurons in layer 1
        activation = torch.sigmoid(self.linear1(x))  # calls linear, then sigmoid
        z2 = torch.sigmoid(self.linear2(activation))  # calls linear on the 2 activations
        return z2
# =============================================================================
#
#
#
# =============================================================================
def train(Y, X, model, optimizer, criterion, epochs=1000):
    cost = []
    total = 0
    
    for epoch in range(epochs):
        total = 0
        for y, x in zip(Y, X):
            
            yhat = model(x)  # forward implicitly called here
            
            loss = criterion(yhat, y.unsqueeze(0))
 
            optimizer.zero_grad()  # Reset the gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the weights
            

            # Cumulative loss
            total += loss.item()
            
            
        
        cost.append(total)

    return cost
# =============================================================================

X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0

# x is 1D
# layer size of 2
# y is 1D
model = Net(1, 2, 1)

criterion = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

cost = train(Y, X, model, optimizer, criterion)



