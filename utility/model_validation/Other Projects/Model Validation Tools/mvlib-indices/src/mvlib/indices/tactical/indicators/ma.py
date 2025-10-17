import numpy as np
import numba as nb

# @nb.njit(fastmath=True)
# def SMA(array: np.ndarray, n_arr: np.ndarray, n_start: int) -> np.ndarray:
#     """
#     Simple Moving Average (SMA) with variable weights.
#     SMA for the first `n_start` points.
#     Admits non-constant weights based on n_arr.

#     Parameters
#     ----------
#     array : numpy.ndarray
#         Target array.
#     n_arr : numpy.ndarray
#         Weight array for each point in the moving average.
#     n_start : int
#         Start of the SMA calculation.

#     Returns
#     -------
#     SMA_arr : numpy.ndarray
#         SMA of the target array.
#     """
#     result = np.empty((array.shape[0]))
#     # Calculate initial points using cumulative sum
#     result[:n_start] = np.cumsum(array[:n_start]) / n_arr[:n_start]
    
#     # Calculate remaining points using rolling window
#     for i in range(n_start, array.shape[0]):
#         window_size = int(n_arr[i])
#         start_idx = max(0, i - window_size + 1)
#         result[i] = np.sum(array[start_idx:i+1]) / window_size
        
#     return result

# @nb.njit(fastmath=True)
# def SMA_with_std(array: np.ndarray, n_arr: np.ndarray, n_start: int) -> np.ndarray:
#     """
#     Simple Moving Average (SMA) with variable weights.
#     Includes standard deviations.
#     SMA for the first `n_start` points.
#     Admits non-constant weights based on n_arr.

#     Parameters
#     ----------
#     array : numpy.ndarray
#         Target array.
#     n_arr : numpy.ndarray
#         Weight array for each point in the moving average.
#     n_start : int
#         Start of the SMA calculation.

#     Returns
#     -------
#     SMA_arr : numpy.ndarray
#         SMA of the target array.
#     STD_arr : numpy.ndarray
#         Running sample standard deviations of the target array.
#     """
#     means = np.empty((array.shape[0]))
#     stds = np.empty((array.shape[0]))
#     # Calculate initial points using cumulative sum
#     means[:n_start] = np.cumsum(array[:n_start]) / n_arr[:n_start]
#     for i in range(n_start):
#         if i == 0:
#             stds[i] = 0
#         stds[i] = np.sqrt(np.sum((array[0:i+1] - means[i])**2) / (n_arr[i]-1)) if n_arr[i] != 1 else 0
    
#     # Calculate remaining points using rolling window
#     for i in range(n_start, array.shape[0]):
#         window_size = int(n_arr[i])
#         start_idx = max(0, i - window_size + 1)
#         means[i] = np.sum(array[start_idx:i+1]) / window_size
#         stds[i] = np.sqrt(np.sum((array[start_idx:i+1] - means[i])**2) / (window_size-1)) if window_size != 1 else 0

#     return means, stds

# The 'right' one
@nb.njit(fastmath=True)
def SMMA(array : np.ndarray, n_arr : np.ndarray, n_start : int) -> np.ndarray:
    """
    Wilder's Smoothing Moving Average (SMMA) as described in specs.
    SMA for the first `n_start` points.
    Admits non-constant weights.

    Parameters
    ----------
    array : numpy.ndarray
        Target array.
    n_arr : numpy.ndarray
        1/weight for the nth SMMA point.
    n_start : int
        Start of the SMMA calculation.

    Returns
    -------
    SMMA_arr : numpy.ndarray
        SMMA of the target array.
    """
    result = np.empty((array.shape[0]))
    result[:n_start] = np.cumsum(array[:n_start]) / n_arr[:n_start]
    for i in range(n_start, array.shape[0]):
        result[i] = (result[i-1] * (n_arr[i]-1) + array[i]) / n_arr[i]
    return result

# # The 'right' one
# @nb.njit(fastmath=True)
# def EWMA(array : np.ndarray, n_arr : np.ndarray, n_start : int) -> np.ndarray:
#     """
#     Exponential Weighted Moving Average (SMMA) as described in specs.
#     SMA for the first `n_start` points.
#     Admits non-constant weights.

#     Parameters
#     ----------
#     array : numpy.ndarray
#         Target array.
#     n_arr : numpy.ndarray
#         1/weight for the nth SMMA point.
#     n_start : int
#         Start of the SMMA calculation.

#     Returns
#     -------
#     SMMA_arr : numpy.ndarray
#         SMMA of the target array.
#     """
#     alpha = 
#     result = np.empty((array.shape[0]))
#     result[:n_start] = np.cumsum(array[:n_start]) / n_arr[:n_start]
#     for i in range(n_start, array.shape[0]):
#         result[i] = (result[i-1] * (n_arr[i]-1) + array[i]) / n_arr[i]
#     return result