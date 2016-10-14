import numpy as np


NEON_PEAKS = [5852.49, 5881.89, 5944.83, 5975.53,
              6030.00, 6074.34, 6096.16, 6143.06,
              6163.59, 6217.28, 6266.49, 6304.79,
              6334.43, 6382.99, 6402.25, 6506.53,
              6532.88, 6598.95, 6678.28, 6717.04]

MERCURY_PEAKS = [3650.15, 4046.56, 4358.33]


ALL_PEAK_ARRAY = np.array(MERCURY_PEAKS + NEON_PEAKS)


def poly_fit(x, y, degree=2):
    x_list = []
    while degree >= 0:
        x_list.append(np.array(x)**degree)
        degree -= 1
    x = np.array(x_list)
    x_t = x.copy()
    x = x.transpose()
    y = np.array([y]).transpose()
    square_matrix = np.linalg.inv(np.dot(x_t, x))
    a = np.dot(np.dot(square_matrix, x_t), y)
    return a


def apply_fit(pixel_number, fit):
    pixels = np.arange(pixel_number)
    a, b, c = fit[0], fit[1], fit[2]
    wavelength_solution = (discriminant(a, b, c, pixels) - b) / (2. * a)
    return wavelength_solution


def discriminant(a, b, c, x):
    return np.sqrt((-4. * a * (-1 * x + c)) + b ** 2.)


def lambda_da(a, b, c, x):
    d = discriminant(a, b, c, x)
    first_term = b / (2. * a**2.)
    second_term = d / a**2.
    third_term = 2. * (c - x) / (a * d)
    return first_term - second_term - third_term


def lambda_db(a, b, c, x):
    d = discriminant(a, b, c, x)
    first_term = b / (2. * a * d)
    second_term = 1. / (2. * a)
    return first_term - second_term


def lambda_dc(a, b, c, x):
    return -1. / discriminant(a, b, c, x)
