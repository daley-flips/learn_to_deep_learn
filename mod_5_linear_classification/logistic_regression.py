import torch
import torch.nn as nn

class logistic_regression(nn.Module):
    def __init__(self, in_size):  # input size
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(in_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x


if __name__ == '__main__':
    
    model= logistic_regression(1)
    # print(list(model.parameters()))
    # print(model)
    
    x = torch.tensor([[1.0], [100]])
    
    yhat = model(x)
    
    print(yhat)
    
    print('\nafter applying the sigmoid function, outputs are between 0 and 1')