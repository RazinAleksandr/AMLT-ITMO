import torch
from torch.utils.data import DataLoader
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.data.datasets import TimeSeriesDataset
from src.utils.base_transforms import BaseTransform


def forecast_lstm(model, test_loader, device):
    model.eval()
    forecasts = []

    with torch.no_grad():
        for data, _ in test_loader:  # We don't need the labels/target here.
            data = data.to(device)
            output = model(data)
            forecasts.extend(output.cpu().numpy())

    return forecasts


def make_forecasts(model, test_df, n_lag, forecast_horizon, device, transform=BaseTransform()):
    # Create a DataLoader for the test set
    test_dataset = TimeSeriesDataset(test_df, n_lag, forecast_horizon, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Assuming a batch size of 1 for simplicity
    
    # Forecast using the DataLoader
    forecasts = forecast_lstm(model, test_loader, device)
    return forecasts


def arima_train_forecast(train, forecast_horizon, order=None):
    history = [x for x in train]
    predictions = list()

    if not order:
        auto_model = auto_arima(history, trace=False)
        order = auto_model.order
    
    model = ARIMA(history, order=order)
    model_fit = model.fit()
    output = model_fit.forecast(steps=forecast_horizon)
        
    predictions.append(output)
    
    return predictions