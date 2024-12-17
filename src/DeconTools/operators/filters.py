from typing import Tuple, Union

import numpy as np
import scipy.fft as fft


def gaussian_fourier_filter(
    shape: Union[int, Tuple(int)], sigma: Union[float, Tuple[float]]
) -> np.ndarray:
    """computes Gaussian filter in Fourier space

    # Example usage
    # data = ... # your input array
    # filtered_data = np.fft.irfftn(np.fft.rfftn(data) *
    # gaussian_filter_fourier(data.shape, sigma=2.0))

    """
    if not isinstance(sigma, tuple):
        sigma = (sigma,) * len(shape)

    # Create frequency grids, using rfftfreq for the last axis
    freq_grids = [fft.fftfreq(n) for n in shape[:-1]]
    freq_grids.append(fft.rfftfreq(shape[-1]))
    mesh_freq = np.meshgrid(*freq_grids, indexing="ij")

    # Compute squared frequency distances
    freq_squared = sum(f**2 for f in mesh_freq)

    # Create Gaussian in Fourier space
    gaussian = np.exp(
        -2
        * (np.pi**2)
        * freq_squared
        * sum(s**2 * f**2 for s, f in zip(sigma, mesh_freq))
    )

    return gaussian
