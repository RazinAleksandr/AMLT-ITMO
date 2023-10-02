import click
import os
import pandas as pd
import warnings
import joblib

from sklearn.preprocessing import MinMaxScaler

from src.utils.ts_transforms import series_to_supervised
warnings.filterwarnings('ignore')


def process_single_file(input_path, output_directory1, output_directory2, scaler_path, lag_input, forecast_horizon, train_val_test):
    # Prepare data for preprocessing
    data = pd.read_csv(input_path, index_col='id', parse_dates=['date'], infer_datetime_format=True)
    data = data.groupby(['date']).agg({'sales':'mean'}).rename(columns={'sales': 'mean_sales'})
    
    # Save preprocessed data
    data.to_csv(output_directory1, index=True)

    # Difference data
    diff_df = data.copy()
    # Create supervised data
    supervised = series_to_supervised(diff_df, lag_input, forecast_horizon)

    # Splitting supervised data into train, validation, and test sets
    train_size = int(len(supervised) * train_val_test[0])
    val_size = int(len(supervised) * train_val_test[1])

    train, intermediate = supervised.iloc[:train_size], supervised.iloc[train_size:]
    val, test = intermediate.iloc[:val_size], intermediate.iloc[val_size:]

    # Scaling - fit on train and transform train, val, and test, to avoid data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index)
    val_scaled = pd.DataFrame(scaler.transform(val), columns=val.columns, index=val.index)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

    # Save preprocessed data
    train_scaled.to_csv(os.path.join(output_directory2, 'train_preprocessed.csv'), index=True)
    val_scaled.to_csv(os.path.join(output_directory2, 'val_preprocessed.csv'), index=True)
    test_scaled.to_csv(os.path.join(output_directory2, 'test_preprocessed.csv'), index=True)

    # Save scaler for inverse transform
    joblib.dump(scaler, scaler_path)


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_directory1", type=click.Path())
@click.argument("output_directory2", type=click.Path())
@click.argument("scaler_path", type=click.Path())
@click.argument("lag_input", type=click.INT)
@click.argument("forecast_horizon", type=click.INT)
@click.option("--train_val_test", type=click.STRING, default="0.6,0.2,0.2")
def main(input_path: str, output_directory1: str, output_directory2: str, scaler_path: str, lag_input: int, forecast_horizon: int, train_val_test: str):
    train_val_test = tuple(map(float, train_val_test.split(',')))
    process_single_file(input_path, output_directory1, output_directory2, scaler_path, lag_input, forecast_horizon, train_val_test)


if __name__ == "__main__":
    main()
