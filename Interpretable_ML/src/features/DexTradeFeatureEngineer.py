from typing import Optional
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class DexTradeFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, window_length: int) -> None:
        """
        Initialize the DexTradeFeatureEngineer.

        Parameters:
        - window_length (int): Length of the rolling window.
        """
        self.window_length = window_length

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DexTradeFeatureEngineer':
        """
        Fit the transformer.

        Parameters:
        - X (pd.DataFrame): The input dataframe.
        - y (pd.Series, optional): Target values. Unused for this transformer.

        Returns:
        - self (DexTradeFeatureEngineer): The fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe by engineering features specific to DEX trades.

        Parameters:
        - X (pd.DataFrame): The input dataframe.

        Returns:
        - pd.DataFrame: The transformed dataframe with engineered features.
        """
        X = X.copy()
        # Time-based features
        X['day_of_week'] = X['timestamp'].dt.dayofweek
        X['hour_of_day'] = X['timestamp'].dt.hour
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)

        # Ratios and Differences
        X['sell_to_buy_ratio'] = X['sell_token_amount_usd'] / X['buy_token_amount_usd']
        X['buy_to_sell_ratio'] = X['buy_token_amount_usd'] / X['sell_token_amount_usd']
        X['sell_buy_difference'] = X['sell_token_amount_usd'] - X['buy_token_amount_usd']
        X['relative_change_sell'] = X['sell_token_amount_usd'].pct_change().fillna(0)
        X['relative_change_buy'] = X['buy_token_amount_usd'].pct_change().fillna(0)

        # Running Statistics
        X['rolling_avg_sell'] = X['sell_token_amount_usd'].rolling(window=self.window_length).mean()
        X['rolling_avg_buy'] = X['buy_token_amount_usd'].rolling(window=self.window_length).mean()
        X['rolling_std_sell'] = X['sell_token_amount_usd'].rolling(window=self.window_length).std()
        X['rolling_std_buy'] = X['buy_token_amount_usd'].rolling(window=self.window_length).std()
        X['ema_sell'] = X['sell_token_amount_usd'].ewm(span=self.window_length, adjust=False).mean()
        X['ema_buy'] = X['buy_token_amount_usd'].ewm(span=self.window_length, adjust=False).mean()
        X.fillna(0, inplace=True)

        # Categorical Features Encoding
        X = pd.get_dummies(X, columns=['sell_token_name', 'buy_token_name', 'exchange_name'], drop_first=True, dtype=float)
        X['both_tokens_verified'] = ((X['buy_token_verified']) & (X['sell_token_verified'])).astype(int)
        X['neither_token_verified'] = ((~X['buy_token_verified']) & (~X['sell_token_verified'])).astype(int)

        # Gas Related Features
        X['gas_ratio_sell'] = X['gas'] / X['sell_token_amount_usd']
        X['gas_ratio_buy'] = X['gas'] / X['buy_token_amount_usd']
        X['max_priority_fee_per_gas_ration_gas'] = X['max_priority_fee_per_gas'] / X['gas']
        X['total_fee'] = X['gas'] + X['max_priority_fee_per_gas']

        # Select only numeric features
        X = X.select_dtypes(include=np.number)
        return X