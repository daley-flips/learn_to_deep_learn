# 1.2
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
def list_to_tensor_and_info_tensor():
    arr = [
        [11, 12, 13],
        [21, 22, 23],
        [31, 32, 33]
        ]
    print(arr, '\n')
    
    t = torch.tensor(arr)
    print(t)
    print(t.shape, '<-- 3X3 matric')
    print(t.ndimension(), '<-- 2 dimesnion')
    print(t.numel(), '<-- 9 total elements')
# =============================================================================
# =============================================================================
def access_first_col():
    # tensor[row, col]
    
    t = torch.tensor(
        [[11, 12, 13],
         [21, 22, 23],
         [31, 32, 33]])
    
    col = t[0:, 0]
    print(col)
    
    # 0: == : so just use :
    col = t[:, 0]
    print(col)
# =============================================================================
# =============================================================================
def element_wise_mult():
    # note, this is different from matrix mult
    
    A = torch.tensor(
        [[0, 1],
         [1, 0]])
    B = torch.tensor(
        [[1, 2],
         [3, 4]])
    
    C = A*B
    print(C)
# =============================================================================
# =============================================================================
def matrix_mult():
    # mm is the dot product of
    # rows in A and cols in B
    # assuming we do mm(A, B)
    
    A = torch.tensor(
        [[0, 1],
         [1, 0]])
    B = torch.tensor(
        [[1, 2],
         [3, 4]])
    
    C = torch.mm(A, B)
    print(C)
# =============================================================================
# =============================================================================


# list_to_tensor_and_info_tensor()
# access_first_col()
# element_wise_mult()
# matrix_mult()