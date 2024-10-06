import torch
# =============================================================================
# Dataset is an abstract class from pytorch
# =============================================================================
from torch.utils.data import Dataset
# =============================================================================
#
#
#
# =============================================================================
# toy_set is a subclass of Dataset
# =============================================================================
class toy_set(Dataset):
# =============================================================================
#
#
#
# =============================================================================
# constructor
# =============================================================================    
    def __init__(self, length=100, transform=None):
        # Create two tensors, x with shape [length, 2] and y with shape [length, 1]
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        
        self.len = length
        self.transform = transform
# =============================================================================
#
#
#
# =============================================================================
# FUN FACT!! when you call arr[0], the python interpreter
# secretly calls __getitem__
# by implementing this function ourselfs, we are overwritting that 
# so [] --> __getitem__
# this also means that we can automatically apply transforms just by indexing
# =============================================================================
    def __getitem__(self, index):
        # Fetch the x and y values at the given index
        sample = self.x[index], self.y[index]
        
        # If a transform is provided, apply it to the sample
        if self.transform:
            sample = self.transform(sample)
        
        return sample
# =============================================================================
#
#
#
# =============================================================================
    def __len__(self):
        # Return the length of the dataset
        return self.len
# =============================================================================
#
#
#
# =============================================================================
# =============================================================================
# example of torch.ones
# =============================================================================
if __name__ == '__main__':
    print()

    print('what torch.ones do?')
    print('it creates a tensor matrix of 1s with the specified row, col')
    print('torch.ones(5 rows, 1 col)')
    print(torch.ones(5, 1), '\n')
    
    print('torch.ones(2, 3)')
    print(torch.ones(2, 3))
    
    print('\nsimilar functionality with torch.zeros:')
    
    print('torch.zeros(6, 4)')
    print(torch.zeros(6, 4))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    