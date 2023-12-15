from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (numpy array): Input features
            y (numpy array): Target labels
        """
        # If X and y are DataFrames or Series, convert them to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Convert data to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # Use torch.long for categorical labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
