from typing import Any
import pandas as pd
import shap


class OfflineAnomalyVisualizer:
    def __init__(self, X: pd.DataFrame, clf: Any) -> None:
        """
        Initialize the OfflineAnomalyVisualizer with data and a trained classifier.

        Parameters:
        - X (pd.DataFrame): The input dataframe.
        - clf (Any): Trained classifier model.
        """
        self.X = X.iloc[:, :-1]
        
        # Initialize the explainer using the classifier and the SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(model=clf, feature_names=X.columns)
        # Compute SHAP values for the data
        self.shap_values = self.explainer(X)

    def show_global(self, plot_type: str = 'bar', max_display: int = 10) -> None:
        """
        Display global visualization of model prediction analysis.

        Parameters:
        - plot_type (str): Type of SHAP visualization plot. Either 'bar' or 'beeswarm'.
        - max_display (int): Maximum number of top features to display.
        """
        print('Plot model prediction analysis on global data')
        if plot_type == 'beeswarm':
            ### SHAP Beeswarm plot
            shap.plots.beeswarm(self.shap_values, show=False, max_display=max_display)
        elif plot_type == 'bar':
            ### Bar plot
            shap.plots.bar(self.shap_values, max_display=max_display)

    def show_local(self, sample_id: int, plot_type: str = 'bar', max_display: int = 10) -> None:
        """
        Display local visualization of model prediction analysis for a specific sample.

        Parameters:
        - sample_id (int): Index of the specific sample for visualization.
        - plot_type (str): Type of SHAP visualization plot. Either 'bar' or 'force_plot'.
        - max_display (int): Maximum number of top features to display.
        """
        print(f'Plot model prediction analysis on {sample_id} trade')
        if plot_type == 'bar':
            shap.plots.bar(self.shap_values[sample_id], max_display=max_display)
        elif plot_type == 'force_plot':
            shap.plots.force(self.shap_values[sample_id], show=True, matplotlib=True)