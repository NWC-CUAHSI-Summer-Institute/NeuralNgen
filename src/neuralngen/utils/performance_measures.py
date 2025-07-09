import numpy as np

def nse(y_true, y_pred):
    """
    Nash-Sutcliffe Efficiency.
    
    Parameters
    ----------
    y_true : np.ndarray, shape [T]
    y_pred : np.ndarray, shape [T]
    
    Returns
    -------
    float
        NSE value.
    """
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - numerator / denominator if denominator > 0 else np.nan