import pandas as pd
import matplotlib.pyplot as plt


def plot_one_step_predictions(data, test_raw, unscaled):
    """
    Plots the model's predictions alongside the actual data.
    
    Parameters:
    - data: DataFrame with the full dataset (train + val + test).
    - test_raw: DataFrame with actual test values.
    - unscaled: DataFrame with predictions.
    """
    fig, ax = plt.subplots(figsize=(15,7))
    
    # Plot Train + Val data
    len_train_val = len(data) - len(test_raw)
    combined_actual = pd.concat([data['mean_sales'][:len_train_val], test_raw['mean_sales']])
    plt.plot(combined_actual.index, combined_actual.values, label='Train + Val', color='blue')

    plt.plot(test_raw.index, test_raw['mean_sales'].values, label='Test Actual', color='green')
    plt.plot(test_raw.index, unscaled['predicted_1'].values, label='Predictions', color='red', linestyle='--')
    
    # Highlight the test data area
    plt.axvspan(test_raw.index[0], test_raw.index[-1], facecolor='lightgray', alpha=0.5)

    # Title, labels, and other plot settings
    plt.title('Model Predictions vs Actual Data')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    return fig, ax

def plot_forecast_horizon(data, test_raw, unscaled, lag_input=None, forecast_horizon=None):
    """
    Plots the model's predictions alongside the actual data.
    
    Parameters:
    - data: DataFrame with the full dataset (train + val + test).
    - test_raw: DataFrame with actual test values.
    - unscaled: DataFrame with predictions.
    - lag_input: used lags for prediction for 'one_point' mode
    - forecast_horizon: forecast horizon for 'one_point' mode
    """
    fig, ax = plt.subplots(figsize=(15,7))
    
    # Plot Train + Val data
    len_train_val = len(data) - len(test_raw)
    combined_actual = pd.concat([data['mean_sales'][:len_train_val], test_raw['mean_sales']])
    plt.plot(combined_actual.index, combined_actual.values, label='Test', color='blue')

    # Take the last test sample and its forecast for the specified horizon
    actual_data = test_raw.iloc[-forecast_horizon:]
    predicted_data = unscaled.iloc[-forecast_horizon, -forecast_horizon:] 
            
    plt.plot(actual_data.index, actual_data.values, label='Test forecast horison', color='green')
    plt.plot(actual_data.index, predicted_data.values, label=f'Predicted (next {forecast_horizon} steps)', color='red', linestyle='--')
    
    # Highlight the test data area
    plt.axvspan(test_raw.index[-forecast_horizon], test_raw.index[-1], facecolor='lightgray', alpha=0.5)

    # Title, labels, and other plot settings
    plt.title('Model Predictions vs Actual Data')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set x-axis limit for zoom
    left_limit = test_raw.index[-forecast_horizon*3]
    right_limit = test_raw.index[-1]
    ax.set_xlim(left_limit, right_limit)

    plt.tight_layout()
    # plt.show()
    return fig, ax


def plot_forecast(real_data, train_ids, val_ids, forecast, forecast_horizon):
    fig, ax = plt.subplots(figsize=(15,7))
    
    # Plot Train + Val data
    plt.plot(train_ids, real_data.loc[train_ids].values, label='Train', color='blue')

    plt.plot(val_ids, real_data.loc[val_ids].values, label='Test forecast horison', color='green')
    plt.plot(val_ids, forecast, label=f'Predicted (next {forecast_horizon} steps)', color='red', linestyle='--')
    
    # Highlight the test data area
    plt.axvspan(val_ids[0], val_ids[-1], facecolor='lightgray', alpha=0.5)

    # Title, labels, and other plot settings
    plt.title('Model Predictions vs Actual Data')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set x-axis limit for zoom
    left_limit = train_ids[-forecast_horizon*2]
    ax.set_xlim(left_limit, val_ids[-1])

    plt.tight_layout()
    # plt.show()
    return fig, ax


def plot_models_forecast(real_data, train_ids, val_ids, arima_forecast, lstm_forecast, forecast_horizon):
    fig, ax = plt.subplots(figsize=(15,7))
    
    # Plot Train + Val data
    plt.plot(train_ids, real_data.loc[train_ids].values, label='Test', color='blue')

    plt.plot(val_ids, real_data.loc[val_ids].values, label='Test forecast horison', color='green')
    plt.plot(val_ids, arima_forecast, label=f'ARIMA predictions', color='red', linestyle='--')
    plt.plot(val_ids, lstm_forecast, label=f'LSTM predictions', color='yellow', linestyle='--')
    
    # Highlight the test data area
    plt.axvspan(val_ids[0], val_ids[-1], facecolor='lightgray', alpha=0.5)

    # Title, labels, and other plot settings
    plt.title('LSTM vs ARIMA on test set forecast 0.2 test')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set x-axis limit for zoom
    left_limit = train_ids[-forecast_horizon*2]
    ax.set_xlim(left_limit, val_ids[-1])

    plt.tight_layout()
    # plt.show()
    return fig, ax