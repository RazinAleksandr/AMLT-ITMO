from typing import Optional, Any
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, lag_input, forecast_horizon, transform: Optional[Any] = None):
        self.dataframe = dataframe
        self.lag_input = lag_input
        self.forecast_horizon = forecast_horizon
        
        self.X = self.dataframe.iloc[:, 0:self.lag_input].values
        self.X = self.X.reshape(self.X.shape[0], 1, self.X.shape[1])  # Adjusted for [samples, timesteps, features]
        self.y = self.dataframe.iloc[:, self.lag_input:self.lag_input+self.forecast_horizon].values

        # Apply any necessary transformations to the data
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_batch = self.X[idx]
        y_batch = self.y[idx]

        # Custom transformation (aggregation with to_tensor())
        if self.transform is not None:
            x_batch = self.transform(x_batch)
            y_batch = self.transform(y_batch)

        return x_batch, y_batch

