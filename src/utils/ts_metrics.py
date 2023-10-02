from sklearn.metrics import r2_score
import numpy as np


def compute_metrics(true_values: np.array, predictions: np.array) -> dict:
    """
    Compute MAPE and R^2 metrics for the given true values and predictions.

    Parameters:
    - true_values: Ground truth (correct) target values.
    - predictions: Estimated target values.

    Returns:
    A dictionary containing the MAPE and R^2 metrics.
    """

    # Ensure numpy arrays and handle possible singularities
    true_values, predictions = np.array(true_values), np.array(predictions)
    non_zero_mask = true_values != 0

    # Compute MAPE only for non-zero true values
    mape = np.mean(np.abs((true_values[non_zero_mask] - predictions[non_zero_mask]) / true_values[non_zero_mask])) * 100

    r2 = r2_score(true_values, predictions)

    return {
        "MAPE": mape,
        "R2": r2
    }


def compute_horizon_metrics(true_values: np.array, predictions: np.array) -> dict:
    """
    Compute MAPE and R^2 metrics for each forecast horizon.

    Parameters:
    - true_values: Ground truth (correct) target values for each forecast horizon.
    - predictions: Estimated target values for each forecast horizon.

    Returns:
    A dictionary containing the MAPE and R^2 metrics for each horizon.
    """
    
    num_horizons = predictions.shape[1]
    metrics = {}
    
    for horizon in range(num_horizons):
        start_idx = horizon
        end_idx = -horizon if horizon != 0 else None
        
        horizon_true = true_values[start_idx:].reshape(-1)
        horizon_pred = predictions[:end_idx, horizon]


        horizon_metrics = compute_metrics(horizon_true, horizon_pred)
        for metric_name, metric_value in horizon_metrics.items():
            metrics[f"{metric_name}_step_{horizon + 1}"] = metric_value

    return metrics
