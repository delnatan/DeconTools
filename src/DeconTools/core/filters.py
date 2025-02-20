from typing import List, Tuple, Union

import numpy as np
import scipy.fft as fft


def create_frequency_meshes(
    shape: Union[int, Tuple[int]],
    pixel_spacing: Union[float, Tuple[float]] = None,
) -> List[np.ndarray]:
    """
    Create frequency meshgrids for FFT operations.

    Args:
        shape: Input shape (single int or tuple)
        pixel_spacing: Optional pixel spacing (single float or tuple)
                      If None, returns unscaled frequencies

    Returns:
        List of frequency meshgrids
    """
    shape = (shape,) if isinstance(shape, int) else shape

    if pixel_spacing is not None:
        pixel_spacing = (
            (pixel_spacing,) * len(shape)
            if isinstance(pixel_spacing, float)
            else pixel_spacing
        )

    # Select appropriate frequency function for each axis
    freq_funcs = [
        (fft.fftfreq, fft.rfftfreq)[i == len(shape) - 1]
        for i in range(len(shape))
    ]

    freq_grids = [
        func(n, d=d) if pixel_spacing is not None else func(n)
        for func, n, d in zip(freq_funcs, shape, pixel_spacing or shape)
    ]

    return np.meshgrid(*freq_grids, indexing="ij")


def compute_gaussian_fourier_filter(
    shape: Union[int, Tuple[int]], sigma: Union[float, Tuple[float]]
) -> np.ndarray:
    """
    Compute Gaussian filter in Fourier space.

    Args:
        shape: Input shape (single int or tuple)
        sigma: Standard deviation(s) of the Gaussian (single float or tuple)

    Returns:
        np.ndarray: Gaussian filter in Fourier space

    Example:
        >>> data = ...  # your input array
        >>> filtered_data = np.fft.irfftn(
        ...     np.fft.rfftn(data) * gaussian_fourier_filter(data.shape, sigma=2.0)
        ... )
    """
    shape = (shape,) if isinstance(shape, int) else shape
    sigma = (sigma,) * len(shape) if isinstance(sigma, (int, float)) else sigma

    freq_mesh = create_frequency_meshes(shape)

    # Sum of squared frequency components weighted by sigma
    freq_sum = sum((s * f) ** 2 for s, f in zip(sigma, freq_mesh))

    return np.exp(-2 * np.pi**2 * freq_sum)


def compute_lanczos_filter(
    shape: Union[int, Tuple[int]],
    freq_cutoff: Union[float, Tuple[float]],
    zoom: Union[float, Tuple[float]],
    p: int = 2,
) -> np.ndarray:
    """
    Compute Lanczos filter in frequency domain.

    This effectively creates a 'box' filter whose size is determined by
    the 'zoom' factor

    """
    shape = (shape,) if isinstance(shape, int) else shape
    ndim = len(shape)

    freq_cutoff = (
        (freq_cutoff,) * len(shape)
        if isinstance(freq_cutoff, float)
        else freq_cutoff
    )

    pixel_spacing = (
        (1 / zoom,) * ndim
        if isinstance(zoom, (float, int))
        else tuple(1 / z for z in zoom)
    )

    mesh_freq_scaled = create_frequency_meshes(shape, pixel_spacing)
    mesh_freq = create_frequency_meshes(shape)

    eps = np.finfo(np.float32).eps

    args1 = [np.pi * f + eps for f in mesh_freq_scaled]
    args2 = [np.pi * fp / fc + eps for fp, fc in zip(mesh_freq, freq_cutoff)]

    sinc = np.prod([np.sin(arg) / arg for arg in args1], axis=0)
    sigma = np.prod([np.sin(arg) / arg for arg in args2], axis=0)

    return np.power(sigma, p) * sinc
