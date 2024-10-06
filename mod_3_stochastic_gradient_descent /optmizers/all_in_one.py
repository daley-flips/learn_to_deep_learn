# =============================================================================
from torch.utils.data import Dataset, DataLoader
import torch

class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = -3 * self.x + 1
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
# =============================================================================
#
#
#
# =============================================================================
import torch.nn as nn
class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out
# =============================================================================
#
#
#
# =============================================================================
# def criterion(yhat, y):
#     return torch.mean((y-yhat)**2)
# only losers actually implement it themselves
criterion = nn.MSELoss()
# =============================================================================
#
#
#
# =============================================================================
ds = Data()  # DataSet
tl = DataLoader(dataset=ds,batch_size=1)  # TrainLoader
# =============================================================================
#
#
#
# =============================================================================
model = LR(1, 1)  # linear regression again lol

from torch import nn, optim
optimizer = optim.SGD(model.parameters(), lr=0.01)

# print(optimizer.state_dict())

for epoch in range(100):
    
    total = 0
    for x, y in tl:

        yhat = model(x)
        loss = criterion(yhat, y)
        

        optimizer.zero_grad()
        
        # Backpropagate the gradients
        loss.backward()
        
        # Update the parameters using optimizer
        optimizer.step()
            # ^ this function just does
                # w.data = w.data - lr * w.grad.data
                # b.data = b.data - lr * b.grad.data
        total += loss.item() 
    print('cost:', total)

       























