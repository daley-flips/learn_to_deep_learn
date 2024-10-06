# 1.3
# taking this course
# https://www.coursera.org/learn/deep-neural-networks-with-pytorch
# print('\ni be learning deep\n')
print()
# =============================================================================
import torch  # the only import you ever need to be awesome
import numpy as np  # and maybe this one is cool too
import pandas as pd  # and ofc pandas
import matplotlib.pyplot as plt
# =============================================================================
# read the method names, should be enough to explain
# =============================================================================
def basic_derivative():
    
    # x = 2
    # requires a gradient
    # meaning to take a derivative
    x = torch.tensor(3.0, requires_grad = True)
    print(x)
    
    # y = x^2
    y = x**2
    print(y)
    
    # take derivative of y        
    y.backward()  # y = x^2 --> y = 2x
    
    # y(2) = 2(3)
    # grad(ient)
    res = x.grad
    # Gradient (x.grad): Stores the computed derivative after calling backward()
    print(res)
# =============================================================================
# =============================================================================
def another_derivative():
    
    x = torch.tensor(2.0, requires_grad=True)
    
    z = x**2 + 2*x + 1
    
    print('z(x) = 9')
    print(z)
    
    print('\nnow take derivative')
    z.backward()
    print()
    
    print('z`(x) = 6')
    print(x.grad)
# =============================================================================
# =============================================================================
def partial_derivatives():
    # see notes on how partials work (so easy)
    
    u = torch.tensor(1.0, requires_grad=True)
    v = torch.tensor(2.0, requires_grad=True)
    
    f = u*v + u**2
    print(f, '\n')  # 1*2 + 1^2 = 2 + 1 = 3
    
    print('take (partial) derivative')
    f.backward()
    
    print(u.grad)  # v+2u = 2+2*1 = 4 (see notes for derivation)
    print(v.grad)  # u = 1

# =============================================================================
# =============================================================================


# basic_derivative()
another_derivative()
# partial_derivatives()