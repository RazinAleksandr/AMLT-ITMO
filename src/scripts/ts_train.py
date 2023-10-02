import click
import os
import pandas as pd
import yaml
import warnings
import joblib
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.utils.ts_helpers import postprocess
from src.utils.ts_plots import plot_one_step_predictions, plot_forecast_horizon
from src.utils.base_transforms import BaseTransform
from src.utils.base_helpers import set_seed
from src.utils.ts_metrics import compute_horizon_metrics, compute_metrics
from src.data.datasets import TimeSeriesDataset
from src.models.train import run_train_loop
from src.models.ts_forecast import make_forecasts
from models.time_series.LSTM import LSTMModel

import mlflow

warnings.filterwarnings('ignore')


def run_train(train_val_test, config_path, prediction_path):
    train = pd.read_csv(os.path.join(train_val_test, "train_preprocessed.csv"), index_col=0)
    val = pd.read_csv(os.path.join(train_val_test, "val_preprocessed.csv"), index_col=0)
    test = pd.read_csv(os.path.join(train_val_test, "test_preprocessed.csv"), index_col=0)

    # Read the configuration file
    with open(config_path, 'r') as config_file:
        config: Dict[str, Any] = yaml.safe_load(config_file)
    
    # Set the seed
    seed = config['exp_parameters'].get('manual_seed', 42)
    set_seed(seed)

    # Create the dataset and dataloader
    train_dataset = TimeSeriesDataset(train, config['model_parameters']['lag_input'], config['model_parameters']['forecast_horizon'], transform=BaseTransform())
    train_loader = DataLoader(train_dataset, batch_size=config['train_parameters']['train_batch_size'], shuffle=False)

    val_dataset = TimeSeriesDataset(val, config['model_parameters']['lag_input'], config['model_parameters']['forecast_horizon'], transform=BaseTransform())
    val_loader = DataLoader(val_dataset, batch_size=config['train_parameters']['train_batch_size'], shuffle=False)

    # Initialize model, loss, and optimizer
    model = LSTMModel(
        input_d=config['model_parameters']['lag_input'], 
        hidden_d=config['lstm_model_parameters']['hidden_d'], 
        layer_d=config['lstm_model_parameters']['layer_d'], 
        output_d=config['model_parameters']['forecast_horizon']
        ).to(config['exp_parameters']['device'])
    
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=config['exp_parameters']['LR'])

    # Start a new MLflow run
    with mlflow.start_run():
        
        # Log parameters from your config
        mlflow.log_params({
            "seed": seed,
            "lag_input": config['model_parameters']['lag_input'],
            "forecast_horizon": config['model_parameters']['forecast_horizon'],
            "hidden_d": config['lstm_model_parameters']['hidden_d'],
            "layer_d": config['lstm_model_parameters']['layer_d'],
            "LR": config['exp_parameters']['LR'],
            "epochs": config['exp_parameters']['epochs'],
        })

        # train model
        train_losses, val_losses = run_train_loop(
            epochs=config['exp_parameters']['epochs'], 
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=config['exp_parameters']['device']
            )
    
        # Log train and val losses. Assuming they are lists, you can log them like this:
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        # test model
        forecasts = make_forecasts(
            model=model, 
            test_df=test, 
            n_lag=config['model_parameters']['lag_input'], 
            forecast_horizon=config['model_parameters']['forecast_horizon'], 
            device=config['exp_parameters']['device']
            )

        # transform to normal scale
        scaler = joblib.load(config['paths']['scaler'])
        preprocessed = pd.read_csv(config['paths']['preprocessed'], index_col=0)
    
        forecased = postprocess(
            forecasts=forecasts, 
            real_data=test, 
            lag_input=config['model_parameters']['lag_input'], 
            scaler=scaler
            )

        # calculate test metrics MAPE and R2
        one_step_test_metrics = compute_horizon_metrics(
            true_values=preprocessed.loc[test.index].values,
            predictions=forecased.iloc[:, -config['model_parameters']['forecast_horizon']:].values
            )
        mlflow.log_metrics(one_step_test_metrics)
        
        forecast_horizon_test_metrics = compute_metrics(
            true_values=preprocessed.loc[test.index].iloc[-config['model_parameters']['forecast_horizon']:].values.reshape(-1), 
            predictions=forecased.iloc[-config['model_parameters']['forecast_horizon'], -config['model_parameters']['forecast_horizon']:].values
        )

        print(forecast_horizon_test_metrics)
        mlflow.log_metrics(forecast_horizon_test_metrics)

        # plot results on test data
        fig, ax = plot_one_step_predictions(
            data=preprocessed, 
            test_raw=preprocessed.loc[test.index], 
            unscaled=forecased, 
            )
        # Log/Save plots
        plot_name = "LSTM_test_one_step_forecast.png"
        plot_path = os.path.join(config['paths']['plots'], plot_name)
        fig.savefig(plot_path)
        mlflow.log_figure(fig, plot_name)

        # plot forecast for test point
        fig, ax = plot_forecast_horizon(
            data=preprocessed, 
            test_raw=preprocessed.loc[test.index], 
            unscaled=forecased, 
            lag_input=config['model_parameters']['lag_input'],
            forecast_horizon=config['model_parameters']['forecast_horizon']
            )
        # Log/Save plots
        plot_name = "LSTM_test_forecast_horizon.png"
        plot_path = os.path.join(config['paths']['plots'], plot_name)
        fig.savefig(plot_path)
        mlflow.log_figure(fig, plot_name)
        
        # Log/Save the model
        model_path = os.path.join(config['paths']['model'], "LSTM_model.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "LSTM_model")

        # Save predictions
        forecased.to_csv(os.path.join(prediction_path, "lstm_prediction.csv"))


@click.command()
@click.argument("train_val_test", type=click.Path(exists=True))
@click.argument("config_path", type=click.Path())
@click.argument("prediction_path", type=click.Path())
def main(train_val_test: str, config_path: str, prediction_path: str):
    run_train(train_val_test, config_path, prediction_path)


if __name__ == "__main__":
    main()
