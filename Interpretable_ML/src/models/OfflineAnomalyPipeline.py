from typing import Union, Optional, Any
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest


class OfflineIsolationForest(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 n_estimators: int, 
                 max_samples: Union[int, str], 
                 bootstrap: bool, 
                 n_jobs: int, 
                 random_state: int, 
                 contamination: float, 
                 column_name: str = 'anomaly_predicted') -> None:
        """
        Initialize the OfflineIsolationForest.

        Parameters:
        - n_estimators (int): Number of base estimators in the ensemble.
        - max_samples (int or str): Number of samples to draw from the training set to train each base estimator.
        - bootstrap (bool): Whether or not to bootstrap the samples.
        - n_jobs (int): Number of CPU cores to use when training.
        - random_state (int): Seed used by the random number generator.
        - contamination (float): Proportion of outliers in the dataset.
        - column_name (str, optional): Name of the column where predictions will be stored. Defaults to 'anomaly_predicted'.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.contamination = contamination
        self.column_name = column_name

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OfflineIsolationForest':
        """
        Fit the transformer using the provided data.

        Parameters:
        - X (pd.DataFrame): The input dataframe.
        - y (pd.Series, optional): Target values. Unused for this transformer.

        Returns:
        - self (OfflineIsolationForest): The fitted transformer.
        """
        self.clf = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            contamination=self.contamination
        )
        self.clf.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using the fitted transformer and store the results in the input dataframe.

        Parameters:
        - X (pd.DataFrame): The input dataframe.

        Returns:
        - pd.DataFrame: The dataframe with an additional column for anomaly predictions.
        """
        X = X.copy()
        X[self.column_name] = self.clf.predict(X)
        return X


class OfflineAnomalyPipeline(Pipeline):
    def fit_predict(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                    **fit_params: Any) -> pd.DataFrame:
        """
        Fits the transformers and final estimator in the pipeline sequentially 
        on the data and then predict anomalies.

        Parameters:
        - X (pd.DataFrame): The input dataframe.
        - y (pd.Series, optional): Target values. Unused for this pipeline.
        - fit_params (Any): Additional parameters to pass to the fit method of the transformers and estimator.

        Returns:
        - pd.DataFrame: The transformed dataframe with an additional column for anomaly predictions.
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].fit(Xt).transform(Xt)