import click
import os
import pandas as pd
import yaml
import warnings
import joblib

from src.utils.base_helpers import set_seed
from src.models.forecast import arima_train_forecast
from src.utils.ts_helpers import arima_postprocess
from src.utils.ts_metrics import compute_metrics
from src.utils.ts_plots import plot_forecast

import mlflow

warnings.filterwarnings('ignore')


def run_train(train_val_test, config_path, prediction_path):
    train = pd.read_csv(os.path.join(train_val_test, "train_preprocessed.csv"), index_col=0)
    val = pd.read_csv(os.path.join(train_val_test, "val_preprocessed.csv"), index_col=0)
    test = pd.read_csv(os.path.join(train_val_test, "test_preprocessed.csv"), index_col=0)
    
    train = pd.concat([train, val], axis=0)
    
    # Read the configuration file
    with open(config_path, 'r') as config_file:
        config: Dict[str, Any] = yaml.safe_load(config_file)
    
    # Set the seed
    seed = config['exp_parameters'].get('manual_seed', 42)
    set_seed(seed)

    # Start a new MLflow run
    with mlflow.start_run():
        
        # Log parameters from your config
        mlflow.log_params({
            "seed": seed,
            "lag_input": config['model_parameters']['lag_input'],
            "forecast_horizon": config['model_parameters']['forecast_horizon'],
            "order": config['arima_model_parameters']['order']
        })

        # train, forecast model
        train = pd.concat([train["var1(t)"], val["var1(t)"]], axis=0)
        train = pd.concat([train, test["var1(t)"].iloc[:-config['model_parameters']['forecast_horizon']]], axis=0)

        forecasted = arima_train_forecast(
            train=train,
            forecast_horizon=config['model_parameters']['forecast_horizon'], 
            order=config['arima_model_parameters']['order']
        )
        
        # transform to normal scale
        scaler = joblib.load(config['paths']['scaler'])
        preprocessed = pd.read_csv(config['paths']['preprocessed'], index_col=0)
    
        forecased = arima_postprocess(
            forecasts=forecasted, 
            real_data=test.iloc[-config['model_parameters']['forecast_horizon']:], 
            lag_input=config['model_parameters']['lag_input'], 
            scaler=scaler
        )

        # calculate test metrics MAPE and R2
        test_metrics = compute_metrics(
            true_values=preprocessed.loc[test.index].iloc[-config['model_parameters']['forecast_horizon']:].values.reshape(-1),
            predictions=forecased.iloc[:, -config['model_parameters']['forecast_horizon']:].values.reshape(-1)
        )
        mlflow.log_metrics(test_metrics)

        # plot results on test data
        fig, ax = plot_forecast(
            real_data=preprocessed, 
            train_ids=train.index, 
            val_ids=test.iloc[-config['model_parameters']['forecast_horizon']:].index, 
            forecast=forecased.iloc[:, -config['model_parameters']['forecast_horizon']:].values.reshape(-1),
            forecast_horizon=config['model_parameters']['forecast_horizon']
        )

        # Log/Save plots
        plot_name = "ARIMA_test_forecast_horizon.png"
        plot_path = os.path.join(config['paths']['plots'], plot_name)
        fig.savefig(plot_path)
        mlflow.log_figure(fig, plot_name)

        # Save predictions
        forecased.to_csv(os.path.join(prediction_path, "arima_prediction.csv"))



@click.command()
@click.argument("train_val_test", type=click.Path(exists=True))
@click.argument("config_path", type=click.Path())
@click.argument("prediction_path", type=click.Path())
def main(train_val_test: str, config_path: str, prediction_path: str):
    run_train(train_val_test, config_path, prediction_path)


if __name__ == "__main__":
    main()
