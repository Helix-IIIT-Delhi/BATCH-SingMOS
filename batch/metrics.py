"""Evaluation Metrics for SingMOS Prediction"""

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, pearson_correlation
import numpy as np


def compute_singmos_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute metrics for SingMOS prediction.

    Args:
        predictions: Predicted SingMOS scores
        targets: Ground truth SingMOS scores

    Returns:
        Dictionary with MAE, RMSE, and Pearson correlation
    """
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)

    # Pearson correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": correlation,
    }
