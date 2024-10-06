# chapter 1.1
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
def create_tensor():
    # creating a 1-D tensor (One-D-Tensor: odt)
    odt = torch.tensor([7, 4, 2, 3, 6])
    print(odt, '\n')
    print('indexing works just as arrays')
    print(odt[0], odt[1], '\n')
# =============================================================================
# =============================================================================
def play_with_types():
    odt = torch.tensor([7, 4, 2, 3, 6])
    print('use .dtype to get data type')
    data_type = odt.dtype
    print(data_type)
    
    
    odt2 = torch.tensor([7.5, 4.5, 2.5, 3.5, 6.65432])
    print(odt2.dtype, '\n')
    odt = torch.tensor([7, 4, 2, 3, 6])
    
    print('use type() to get tensor type')
    print(odt.type())
    print(odt2.type(), '\n')
    
    print('can also explicitly set the type')
    
    float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
    print(float_tensor)
    print('notice the decimals added \n')
    
    print('can also change tensor types')
    print(odt.type())
    odt = odt.type(torch.FloatTensor)
    print(odt.type(), '\n')
# =============================================================================
# =============================================================================
def dimensionality():
    t = torch.tensor([1, 2, 3, 4, 5])
    print(t, '\n')
    
    print('.size() gives length?')
    print(t.size(), '\n')
    
    print('.ndimension() prints how many dimensions')
    print(t.ndimension(), '\n')
    
    print('.view() can change the dimensionality/ add columns')
    # per my 5 rows, add 1 column
    # t = t.view(5,1)
    # or just use -1 and it'll count the row number
    t = t.view(-1,1)
    print(t)
    print(t.ndimension())
# =============================================================================
# =============================================================================
def np_to_torch_to_np():
    # the idea here is that any torch can be a np arr
    # and any np arr can be a torch 
    # easy
    arr = np.array([0.1, 0.5, 1, 65])
    
    print(arr, '<-- nice lil numpy arr\n')
    print('wanna convert to tensor?')
    t = torch.from_numpy(arr)
    print('torch.from_numpy(NP ARRAY GO HERE)')
    print(t, '<-- boom\n')
    
    print('wanna convert back to np?')
    print('use .numpy()')
    t = t.numpy()
    print(t, '<-- woo!')
# =============================================================================
# =============================================================================
def pandas_to_torch():
    # just go pandas --> np, then np --> torch
    df = pd.Series([0.1, 1.1, 10.1, 89])
    print(df, '<-- pandas df\n')
    
    print('convert to np:')
    print('use .values')
    arr = df.values
    print(arr, '<-- np arr\n')
    
    # same method from before
    print('np == torch :)')
    t = torch.from_numpy(arr)
    print(t, '<-- yipee')
# =============================================================================
# =============================================================================
def list_and_primitives():
    t = torch.tensor([1, 2, 3, 4, 5])
    print(t, '\n')
    print('want a regular python list?')
    print('use .tolist()')
    regular_list = t.tolist()
    print(regular_list, '\n')
    
    print('reminder that each element in the tensor is a tensor')
    print(t[0], t[1], '\n')
    print('what if we want a python number?')
    print('use .item()')
    print(t[0].item(), t[1].item())
# =============================================================================
# =============================================================================
def vector_add_sub_mul():
    u = torch.tensor([0, 1])
    v = torch.tensor([2, 4])
    print('u', u)
    print('v', v, '\n')
    
    z = u+v
    print('add u+v')
    print(z, '\n')
    
    z = u-v
    print('subtract u-v')
    print(z, '\n')
    
    z = u*v
    print('u*v is entry-wise product, just multiples the corresponding indexes together')
    print(z, '\n')
    
    z = 2*v
    print('2*v will multiply every value in the vector by 2')
    print(z, '\n')
    
    z = v+1
    print('add v+1 adds 1 to every value in the array')
    print(z, '\n')
# =============================================================================
# =============================================================================
def dot_product():
    # think of dot product to show how similar vectors are
    # 0 --> theyre perpendicular and not similar
    # 10000000 --> they are very similar 
    u = torch.tensor([0, 1])
    v = torch.tensor([2, 4])
    print('u', u)
    print('v', v, '\n')
    
    print('use torch.dot(u, v)')
    similarity = torch.dot(u,v)
    print(similarity)
# =============================================================================
# =============================================================================
def evenly_distributed_nums():
    # idk i just think this function is cool
    print('lets get 41 evenly spaced numbers between 5 and 25')
    
    t = torch.linspace(5, 25, 41)
    
    print(t)
# =============================================================================
# =============================================================================
def cool_sin_graph():
    
    # create 100 evenly spaced nums from 0 to 2pi
    x = torch.linspace(0, 2*np.pi, 100)
    
    # apply y = sin(x)
    y = torch.sin(x)
    
    plt.plot(x, y)


def cool_cos_graph():
    
    # create 100 evenly spaced nums from 0 to 2pi
    x = torch.linspace(0, 2*np.pi, 100)
    
    # apply y = sin(x)
    y = torch.cos(x)
    
    plt.plot(x, y)
# =============================================================================
# =============================================================================


# create_tensor()
# play_with_types()
# dimensionality()
# np_to_torch_to_np()  # fluid between np and torch
# pandas_to_torch()
# list_and_primitives()
# vector_add_sub_mul()
# dot_product()
# evenly_distributed_nums()
# cool_sin_graph()
# cool_cos_graph()



