import click
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.utils.ts_transforms import series_to_supervised


def process_single_file(input_path, output_directory, lag_input, forecast_horizon, test_size):
    # prepare data for preprocessing
    data = pd.read_csv(input_path, index_col='id', parse_dates=['date'], infer_datetime_format=True)
    data = data.groupby(['date']).agg({'sales':'mean'}).rename(columns={'sales': 'mean_sales'})

    # Difference data
    diff_df = data.diff().dropna()

    # Create supervised data
    supervised = series_to_supervised(diff_df, lag_input, forecast_horizon)

    # Train-test split using train_test_split
    train, test = train_test_split(supervised, test_size=test_size, shuffle=False)

    # Scaling - fit on train and transform train and test, to avoid data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

    # Save preprocessed data (you can add this to DVC later)
    train_scaled.to_csv(os.path.join(output_directory, 'train_preprocessed.csv'))
    test_scaled.to_csv(os.path.join(output_directory, 'test_preprocessed.csv'))


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
@click.argument("lag_input", type=click.INT)
@click.argument("forecast_horizon", type=click.INT)
@click.option("--test_size", type=click.FLOAT, default=0.2)
def main(input_path: str, output_directory: str, lag_input: int, forecast_horizon: int, test_size: float):
    process_single_file(input_path, output_directory, lag_input, forecast_horizon, test_size)


if __name__ == "__main__":
    main()
