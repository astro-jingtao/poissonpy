import numpy as np

from sympy import lambdify, diff
from sympy.abc import x, y
from ait.conv import convolve, conv_cube_2d


# Function related
def get_sp_function(expr):
    return lambdify([x, y], expr, "numpy")


def get_sp_laplacian_expr(expr):
    return diff(expr, x, 2) + diff(expr, y, 2)


def get_sp_derivative_expr(expr, var):
    return diff(expr, var, 1)


def conv(arr, kernel, batch=False, boundary='reflect', padding=False):
    if batch:
        return conv_cube_2d(arr,
                            kernel,
                            method='fft',
                            vel_axis=-1,
                            pad=True,
                            pad_boundary=boundary,
                            normalize_kernel=False,
                            nan_treatment='fill')
    else:
        return convolve(arr,
                        kernel,
                        method='fft',
                        boundary=boundary,
                        pad=padding,
                        normalize_kernel=False,
                        nan_treatment='fill')


def get_np_gradient(arr,
                    dx=1,
                    dy=1,
                    forward=True,
                    batch=False,
                    boundary='reflect',
                    padding=False):

    if forward:
        kx = np.array([[0, 0, 0], [0, -1 / dx, 1 / dx], [0, 0, 0]])
        ky = np.array([[0, 0, 0], [0, -1 / dy, 0], [0, 1 / dy, 0]])
    else:
        kx = np.array([[0, 0, 0], [-1 / dx, 1 / dx, 0], [0, 0, 0]])
        ky = np.array([[0, -1 / dy, 0], [0, 1 / dy, 0], [0, 0, 0]])

    Gx = conv(arr, kx, boundary=boundary, padding=padding, batch=batch)
    Gy = conv(arr, ky, boundary=boundary, padding=padding, batch=batch)

    return Gx, Gy


def get_np_laplacian(arr,
                     dx=1,
                     dy=1,
                     batch=False,
                     boundary='reflect',
                     padding=False):
    kernel = np.array([[0, 1 / (dy**2), 0],
                       [1 / (dx**2), -2 / (dx**2) - 2 / (dy**2), 1 / (dx**2)],
                       [0, 1 / (dy**2), 0]])

    return conv(arr, kernel, boundary=boundary, padding=padding, batch=batch)


def get_np_div(gx, gy, dx=1, dy=1):
    gxx, _ = get_np_gradient(gx, dx, dy, forward=False)
    _, gyy = get_np_gradient(gy, dx, dy, forward=False)
    return gxx + gyy


def get_np_gradient_amplitude(gx, gy):
    return np.sqrt(gx**2 + gy**2)
