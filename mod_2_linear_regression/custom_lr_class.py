import torch
import torch.nn as nn
# =============================================================================
# LR is a subclass of nn.Module
class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
# =============================================================================
# the forward method is interesting because it is inplicitly called
# yhat = model(x) calls forward
# any LR object followed by a () will call it
# =============================================================================
    def forward(self, x):
        out = self.linear(x)
        return out
# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    # Create a linear regression model with 1 input and 1 output
    model = LR(1, 1)
    
    # Print model parameters (weights and bias)
    print(list(model.parameters()), '\n')
    
    # Input tensor
    x = torch.tensor([1.0])
    
    # Forward pass through the model
    yhat = model(x)
    
    # Output prediction
    print('yhat', yhat, '\n')
    
    print("Python dictionary:", model.state_dict(), '\n')
    print("keys: ", model.state_dict().keys())
    print("values: ", model.state_dict().values())

