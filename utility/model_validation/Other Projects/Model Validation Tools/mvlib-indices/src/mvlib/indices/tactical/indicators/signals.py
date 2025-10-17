import numpy as np
from typing import Any
from .base import Signal

def fill_zeros_with_last(arr : np.ndarray) -> np.ndarray:
    """
    Fill zeroes in array with last non-zero value before the zeroes. Does not affect leading zeros (zeros before any non-zero value).

    Parameters
    ----------
    arr : np.ndarray
        Array to remove 0s from.

    Returns
    -------
    nn_arr : np.ndarray
        Non-zero array (except for leading zeros).
    """
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]

class Discrete_Const(Signal):
    """
    Discrete signal for an indicator with constant thresholds, representing a positive, negative or zero only.
    This does not return the leverage.

    ``` 
    if x > upper_threshold:
        return 1
    if x < lower_threshold:
        return -1
    else:
        return 0 
    ```

    Attributes
    ----------
    upper_threshold : float
        Upper threshold (overbought). Typically above 50.
    lower_threshold : float
        Lower threshold (oversold). Typically below 50.
    """
    def __init__(self, upper_threshold : float, lower_threshold : float) -> None:
        """
        Parameters
        ----------
        upper_threshold : float
            Upper threshold (overbought). Typically above 50.
        lower_threshold : float
            Lower threshold (oversold). Typically below 50.
        """
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
    
    def __call__(self, value : np.ndarray) -> Any:
        return np.where(value < self.lower_threshold, -1,
                        np.where(value > self.upper_threshold, 1,
                                    0))
    
class Discrete_Var(Signal):
    """
    Discrete signal for an indicator with variable thresholds, representing a positive, negative or zero only.
    This does not return the leverage.

    ``` 
    if x > variable_upper_threshold:
        return 1
    if x < variable_lower_threshold:
        return -1
    else:
        return 0 
    ```
    """
    def __init__(self) -> None:
        pass
    
    def __call__(self, value : np.ndarray, upper_thresholds : np.ndarray, lower_thresholds : np.ndarray) -> Any:
        return np.where(value < lower_thresholds, -1,
                        np.where(value > upper_thresholds, 1,
                                    0))
