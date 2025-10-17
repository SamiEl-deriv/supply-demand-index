import numpy as np


def linear_interpolate(points: np.ndarray, x: float) -> float:
    """
    Given an array consisting of 2 arrays of form [x,y], perform bilinear interpolation with a given x within the line segment formed by the 2 points.

    Parameters
    ----------
    points : numpy.ndarray
        A 2 x 2 array containing (x,y)-coordinates representing the 2 corners of the line segment & y-values to interpolate against
    x : float
        The requested x-value to interpolate for

    Returns
    -------
    float
        The interpolated y-value corresponding to the provided x-ordinates
    """

    # Requires exactly 4 points
    if points.shape[0] != 2:
        raise ValueError(
            f"points requires exactly 2 points. was given {points.shape[0]} points")

    # sort first column to ascending
    sorted_points = points[np.argsort(points[:, 0])]

    # Extract values
    x_vals = sorted_points[:, 0]
    y_vals = sorted_points[:, 1]

    # Çheck if x-ordinates are unique
    if points.shape[0] != np.unique(x_vals, axis=0).shape[0]:
        raise ValueError(f"Requires distinct x-ordinates")

    width = x_vals[1] - x_vals[0]
    result = 0

    # Interpolate as weighted average
    for i in range(2):
        result += y_vals[i] * (x_vals[i - 1] - x)

    return result / width


def tv_interpolate(points: np.ndarray, t: float):
    """
    Given an array consisting of 2 (t,s)-points, perform total variatiom interpolation with a given t.

    Example
    -------

    >>> from .utils import bilinear_interpolation
    >>> import numpy as np
    >>> points = np.array([[0,0], [0,1]])
    1.0

    Parameters
    ----------
    points : numpy.ndarray
        An array consisting of 2 (t,s)-points
    t : float
        The requested t-value to interpolate for

    Returns
    -------
    float
        The total variance interpolated s-value corresponding to the provided t-ordinate
    """

    # Requires exactly 4 points
    if points.shape[0] != 2:
        raise ValueError(
            f"points requires exactly 2 points. was given {points.shape[0]} points")

    # sort first column to ascending
    sorted_points = points[np.argsort(points[:, 0])]

    # Extract values
    t_vals = sorted_points[:, 0]
    s_vals = sorted_points[:, 1]

    # Çheck if x-ordinates are unique
    if points.shape[0] != np.unique(t_vals, axis=0).shape[0]:
        if points.shape[1] != np.unique(s_vals, axis=0).shape[0]:
            raise ValueError(
                f"Invalid coordinates, both have the same t-ordinates but different s-ordinates")
        else:
            return s_vals[0]

    tv = 0

    for i in range(2):
        tv += abs(t - t_vals[i]) / (t_vals[1] - t_vals[0]) * \
            t_vals[i - 1] * (s_vals[i - 1]**2)

    return np.sqrt(tv / t)
