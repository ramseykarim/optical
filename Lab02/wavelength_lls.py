import numpy as np


NEON_PEAKS = [5852.49, 5881.89, 5944.83, 5975.53,
              6030.00, 6074.34, 6096.16, 6143.06,
              6163.59, 6217.28, 6266.49, 6304.79,
              6334.43, 6382.99, 6402.25, 6506.53,
              6532.88, 6598.95, 6678.28, 6717.04]

MERCURY_PEAKS = [3650.15, 4046.56, 4358.33]


ALL_PEAK_ARRAY = np.array(NEON_PEAKS + MERCURY_PEAKS)


def poly_fit(x, y, degree=2):
    x_list = []
    d = degree
    while degree >= 0:
        x_list.append(np.array(x)**degree)
        degree -= 1
    n = x[0].size
    x = np.array(x_list)
    x_t = x.copy()
    x = x.transpose()
    y = np.array([y]).transpose()
    square_matrix = np.linalg.inv(np.dot(x_t, x))
    a = np.dot(np.dot(square_matrix, x_t), y)
    return a

