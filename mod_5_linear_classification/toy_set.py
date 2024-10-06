import torch
from torch.utils.data import Dataset

class toy_set(Dataset):
    # Constructor
    def __init__(self, length=100, transform=None):
        # Generate random 1D feature vectors (x) of shape [length, 1]
        self.x = torch.randn(length, 1)  # Random input features with shape [length, 1]
        
        # Create binary labels (y) using a linear combination with added noise
        self.y = torch.sigmoid(2 * self.x + 0.5 * torch.randn(length, 1))  # Logistic relation
        
        # Binarize the labels: values > 0.5 are class 1, else class 0
        self.y = (self.y > 0.5).float()  # Ensure binary labels
        
        self.len = length
        self.transform = transform
    
    # Get item at a particular index
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        
        # If a transform is provided, apply it to the sample
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    # Return the length of the dataset
    def __len__(self):
        return self.len
