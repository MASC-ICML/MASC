import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch_utils import torch_temporary_seed

class MSCDataset(Dataset):
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 budget: pd.Series):
        """
        Initialize the dataset.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            x_columns (list): List of column names for features (X)
            y_column (str): Column name for target variable (y)
            budget_transform (str): String representation of the budget calculation
        """
        assert X.shape[0] == y.shape[0] == budget.shape[0], "X, y and budget must have the same number of rows"
        self.X = torch.Tensor(X.values)
        self.y = torch.Tensor(y.values)
        self.budget = torch.Tensor(budget.values)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y': self.y[idx],
            'budget': self.budget[idx],
            'index': idx
        }

def create_dataloaders(dataset: MSCDataset,
                       batch_size: int,
                       test_ratio=0.2,
                       val_ratio=0.1,
                       data_seed: int = 42):
    assert 0.0 < test_ratio < 1.0, "Test ratio must be between 0 and 1"
    assert 0.0 < val_ratio < 1.0, "Validation ratio must be between 0 and 1"
    assert test_ratio + val_ratio < 1.0, "Sum of test and validation ratios must be less than 1"

    n = len(dataset)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    with torch_temporary_seed(data_seed):
        indices = torch.randperm(n)

        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size + val_size]
        train_idx = indices[test_size + val_size:]

        generator = torch.Generator()
        generator.manual_seed(data_seed)

        train_sampler = SubsetRandomSampler(train_idx, generator=generator)
        val_sampler = SubsetRandomSampler(val_idx, generator=generator)
        test_sampler = SubsetRandomSampler(test_idx, generator=generator)

        dl_train = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              drop_last=True)
        dl_val = DataLoader(dataset,
                            sampler=val_sampler,
                            batch_size=len(val_idx))
        dl_test = DataLoader(dataset,
                             sampler=test_sampler,
                             batch_size=len(test_idx))

    return dl_train, dl_val, dl_test