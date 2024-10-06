
# 2.1
# taking this course
# https://www.coursera.org/learn/deep-neural-networks-with-pytorch
# print('\ni be learning deep\n')
print()
# =============================================================================
import torch  # the only import you ever need to be awesome
# =============================================================================
# =============================================================================
def basic_linear_eq():
    # let w = 2
    # let b = -1
    
    
    w = torch.tensor(2.0, requires_grad=True)  # grad = True because we have to learn these
    b = torch.tensor(-1.0, requires_grad=True)  # grad = True because we have to learn these
    
    print('w:', w)
    print('b:', b, '\n')
    
    
    def forward(x):
        y = w*x+b
        return y
    
    # let x = 1
    x = torch.tensor([[1],[2]])
    
    yhat = forward(x)
    print('yhat:', yhat)
# =============================================================================
# =============================================================================
from torch.nn import Linear
# ^ need this import
def one_in_one_out():
    torch.manual_seed(1)  # set random seed
    model = Linear(in_features=1, out_features=1)
    
    print('printing parametes, first is slope, then bias:\n')
    print(list(model.parameters()), '\n')
    
    x = torch.tensor([0.0])
    
    yhat = model(x)
    print('yhat:', yhat)
# =============================================================================
# =============================================================================
def vector_in_vector_out():
    torch.manual_seed(1)  # set random seed
    model = Linear(in_features=1, out_features=1)
    
    x = torch.tensor([[1.0],[2.0]]) 
    yhat = model(x)
    print('yhat:', yhat)
# =============================================================================
# =============================================================================



# basic_linear_eq()
# one_in_one_out()
vector_in_vector_out()











































































    
