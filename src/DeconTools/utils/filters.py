from typing import List, Tuple, Union

import torch
from torch.fft import fftfreq, rfftfreq


def create_frequency_meshes(
    shape: Union[int, Tuple[int, ...]],
    pixel_spacing: Union[float, Tuple[float, ...]] = 1.0,
    device=None,
    dtype=torch.float32,
) -> List[torch.Tensor]:
    """
    Create frequency meshgrids for FFT operations using PyTorch.

    Args:
        shape: (int or tuple of ints)
        pixel_spacing: (float or tuple of floats) Optional; if given, scales frequencies.

    Returns:
        List[torch.Tensor]: Meshgrids for each frequency axis.
    """
    shape = (shape,) if isinstance(shape, int) else shape

    if isinstance(pixel_spacing, float):
        pixel_spacing = (pixel_spacing,) * len(shape)

    freq_vecs = []

    for i, n in enumerate(shape):
        d = pixel_spacing[i]
        if i == len(shape) - 1:  # use rfftfreq for last axis
            vec = rfftfreq(n, d, device=device, dtype=dtype)
        else:
            vec = fftfreq(n, d, device=device, dtype=dtype)
        freq_vecs.append(vec)
    # Create meshgrid with 'ij' indexing so that the order matches the input shape
    return list(torch.meshgrid(*freq_vecs, indexing="ij"))


def compute_gaussian_fourier_filter(
    shape: Union[int, Tuple[int, ...]],
    sigma: Union[float, Tuple[float, ...]],
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Compute a Gaussian filter in Fourier space using PyTorch.

    Args:
        shape: (int or tuple of ints)
        sigma: (float or tuple of floats) Standard deviation(s) of the Gaussian.

    Returns:
        torch.Tensor: Gaussian filter in Fourier space.
    """
    shape = (shape,) if isinstance(shape, int) else shape
    sigma = (sigma,) * len(shape) if isinstance(sigma, (int, float)) else sigma

    freq_mesh = create_frequency_meshes(shape, device=device, dtype=dtype)
    # Compute sum_{dim} (sigma * frequency)^2
    freq_sum = sum((s * f) ** 2 for s, f in zip(sigma, freq_mesh))
    return torch.exp(-2 * (torch.pi**2) * freq_sum)


def compute_lanczos_filter(
    shape: Union[int, Tuple[int, ...]],
    freq_cutoff: Union[float, Tuple[float, ...]],
    zoom: Union[float, Tuple[float, ...]],
    p: int = 2,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Compute a Lanczos filter in the Fourier domain using PyTorch.

    Args:
        shape: (int or tuple of ints)
        freq_cutoff: (float or tuple of floats) Frequency cutoff(s).
        zoom: (float or tuple of floats) Zoom factor(s).
        p: Exponent factor.

    Returns:
        torch.Tensor: Lanczos filter in Fourier space.
    """
    shape = (shape,) if isinstance(shape, int) else shape
    ndim = len(shape)
    freq_cutoff = (
        (freq_cutoff,) * ndim
        if isinstance(freq_cutoff, float)
        else freq_cutoff
    )
    # Calculate pixel spacing based on the zoom factor
    if isinstance(zoom, (float, int)):
        pixel_spacing = (1.0 / zoom,) * ndim
    else:
        pixel_spacing = tuple(1.0 / z for z in zoom)

    mesh_freq_scaled = create_frequency_meshes(
        shape, pixel_spacing, device=device, dtype=dtype
    )
    mesh_freq = create_frequency_meshes(shape, device=device, dtype=dtype)
    eps = torch.finfo(dtype).eps

    # Compute arguments for sinc functions
    args1 = [torch.pi * f + eps for f in mesh_freq_scaled]
    args2 = [
        (torch.pi * fp / fc) + eps for fp, fc in zip(mesh_freq, freq_cutoff)
    ]

    # Compute elementwise products across dimensions
    sinc = torch.prod(
        torch.stack([torch.sin(arg) / arg for arg in args1], dim=0), dim=0
    )
    sigma_val = torch.prod(
        torch.stack([torch.sin(arg) / arg for arg in args2], dim=0), dim=0
    )
    return (sigma_val**p) * sinc
