from typing import Optional, List
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class DexTradePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None) -> None:
        """
        Initialize the DexTradePreprocessor.

        Parameters:
        - columns (List[str], optional): Columns to select. If None, default columns will be used.
        """
        self.columns = columns if columns else [
            'timestamp',
            'sell_token_amount_usd',
            'buy_token_amount_usd',
            'sell_token_name',
            'buy_token_name',
            'exchange_name',
            'buy_token_verified',
            'sell_token_verified',
            'gas',
            'max_priority_fee_per_gas'
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DexTradePreprocessor':
        """
        Fit the transformer.

        Parameters:
        - X (pd.DataFrame): The input dataframe.
        - y (pd.Series, optional): Target values. Unused for this transformer.

        Returns:
        - self (DexTradePreprocessor): The fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe by applying preprocessing specific to DEX trades.

        Parameters:
        - X (pd.DataFrame): The input dataframe.

        Returns:
        - pd.DataFrame: The transformed dataframe.
        """
        X = X.copy()
        X.drop_duplicates(inplace=True)
        X = X[self.columns]
        X['timestamp'] = pd.to_datetime(X['timestamp'])
        X.dropna(inplace=True)
        X.sort_values(by='timestamp', ascending=True, inplace=True)
        X.reset_index(drop=True, inplace=True)
        return X