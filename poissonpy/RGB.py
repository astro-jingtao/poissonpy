import numpy as np
from .solvers import Poisson2DRectangle, Poisson2DRegion


def solve_RGB_region(mask, laplacian, boundary_img):

    laplacian = np.asarray(laplacian, dtype=np.float64)
    boundary_img = np.asarray(boundary_img, dtype=np.float64)

    if (laplacian.ndim != 3) or (laplacian.shape[-1] != 3):
        raise ValueError(
            "Interior must be a 3D array with the last dimension of size 3.")

    if (boundary_img.ndim != 3) or (boundary_img.shape[-1] != 3):
        raise ValueError(
            "Boundary must be a 3D array with the last dimension of size 3.")

    all_channels = []

    for i in range(3):
        solver = Poisson2DRegion(mask, laplacian[..., i], boundary_img[..., i])
        solution = solver.solve()
        solution[mask == 0] = boundary_img[mask == 0, i]
        all_channels.append(solution)

    return np.stack(all_channels, axis=-1)