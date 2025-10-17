import numpy as np


def tridiagonal_solver(mat: np.ndarray, d: np.ndarray):
    """
    Solves a tridiagonal system in O(n) time. In particular, it solves

    Ax = d

    for x, where A is a tridiagonal square matrix with non-zero entries only in the diagonal, subdiagonal and superdiagonal.

    Known to be stable if matrix is diagonally dominant or symmetric postiive definite,
    Also known as the Thomas algorithm,

    refs: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
          Theorem 9.12, Nicholas J. Higham (2002). Accuracy and Stability of Numerical Algorithms: Second Edition. SIAM. p. 175

    TODO : Replace mat with 3 diagonals (sub, main and super)

    Parameters
    ----------
    mat : numpy.ndarray
        Tridiagonal matrix (A in equation above)
    d : numpy.ndarray
        Vector of same column dimension as mat

    Returns
    -------
    numpy.ndarray
        The solution vector x
    """

    dimX = mat.shape[0]
    dimY = mat.shape[1]

    assert dimX == dimY, "mat must be a square matrix"

    a = np.diagonal(mat, offset=-1)
    b = np.diagonal(mat).copy()
    c = np.diagonal(mat, offset=1)

    # Forward sweep
    c_prime = np.zeros(c.shape[0])
    d_prime = np.zeros(dimY)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, c_prime.shape[0]):
        c_prime[i] = c[i] / (b[i] - a[i - 1] * c_prime[i - 1])

    for i in range(1, d_prime.shape[0]):
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / \
            (b[i] - a[i - 1] * c_prime[i - 1])

    result = np.zeros(dimY)
    result[-1] = d_prime[-1]
    for i in range(dimY - 2, -1, -1):
        result[i] = d_prime[i] - c_prime[i] * result[i + 1]

    return result


def bilinear_interpolate(corners: np.ndarray, x: float, y: float) -> float:
    """
    Given an array consisting of 4 arrays of form [x,y,z], perform bilinear interpolation with a given (x,y) within the square formed by the 4 points.

    Example
    -------

    >>> from .utils import bilinear_interpolation
    >>> import numpy as np
    >>> points = np.array([[0,0,0], [0,1,1],
                       [1,0,1], [1,1,2]])
    1.0

    Parameters
    ----------
    corners : numpy.ndarray
        A 4 x 3 array containing (x,y,z)-coordinates representing the 4 corners of the square & z-values to interpolate against
    x : float
        The requested x-value to interpolate
    y : float
        The requested t-value to interpolate

    Returns
    -------
    float
        The interpolated z-value corresponding to the provided (x,y)-coordinates
    """

    # Requires exactly 4 points
    if corners.shape[0] != 4:
        raise ValueError(
            f"corners requires exactly 4 points. was given {corners.shape[0]} points")

    coords = corners[:, :2]  # array of x-y coordinates

    # Çheck if (x,y) coordinates are unique
    if corners.shape[0] != np.unique(coords, axis=0).shape[0]:
        raise ValueError(f"Requires distinct (x,y) coordinates")

    # Check that there are precisely 2 distinct x-values and y-values
    if np.unique(coords[:, 0]).shape[0] != 2 or np.unique(
            coords[:, 1]).shape[0] != 2:
        raise ValueError(f"points do not form a square")

    # lexicographical sorting & square construction
    ind = np.lexsort((coords[:, 1], coords[:, 0]))
    square = np.array([corners[i, :] for i in ind]).reshape(2, 2, 3)

    # Extract values
    x_grid = square[:, 0, 0]
    y_grid = square[0, :, 1]
    z_grid = square[:, :, 2]

    # Out of bounds error
    if x < x_grid[0] or x > x_grid[1]:
        raise ValueError(f"x-value to interpolate is out of bounds, {x=}")
    if y < y_grid[0] or y > y_grid[1]:
        raise ValueError(f"y-value to interpolate is out of bounds, {y=}")

    width = x_grid[1] - x_grid[0]
    height = y_grid[1] - y_grid[0]
    result = 0

    # Interpolate
    for i in range(2):
        for j in range(2):
            # Add weighted contribution of each corner
            result += z_grid[i, j] * \
                abs((x_grid[i - 1] - x) * (y_grid[j - 1] - y))

    return result / (width * height)
