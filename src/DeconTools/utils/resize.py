"""
Notes:
Default argument for 'norm' in torch.fft.rfftn is "backward" (no normalization)
Default argument for 'norm' in torch.fft.irfftn is "backward", scaling by 1/n

So if the output is larger then we need to compensate for the intensities
being spread out across more pixel by scaling the coefficients by N_out / N_in.

"""

from typing import List, Tuple

import torch
import torch.fft as fft


def compute_rfftn_size(*shape):
    n = len(shape)
    return tuple(
        s if (i < n) else (s // 2 + 1) for i, s in enumerate(shape, start=1)
    )


def freq_index(n, half=False):
    """computes frequency index

    for complex FFT

    even n :  0 ... N/2 - 1, -N/2 ... -1
                 +                 -
    odd n  :  0 ... (N - 1)/2, (N-1)/2 ... -1
                 +                      -
    """
    if half:
        # real fft indices
        return torch.arange(0, n // 2 + 1)
    else:
        # complex fft indices
        if n % 2 == 0:
            return torch.cat(
                [torch.arange(0, n // 2), torch.arange(-n // 2, 0)]
            )
        else:
            return torch.cat(
                [
                    torch.arange(0, (n - 1) // 2 + 1),
                    torch.arange(-(n - 1) // 2, 0),
                ]
            )


def compute_rfftn_indices(*shape):
    """return broadcasted indices for real FFT tensor

    broadcasting is done by 'rolling' -1 across dims,
    so for len(shape) = 3
    (-1, 1, 1)
    (1, -1, 1)
    (1, 1, -1)


    """
    ndim = len(shape)
    vec = [
        freq_index(s, half=(i == ndim)) for i, s in enumerate(shape, start=1)
    ]
    broadcasted = [
        v.reshape([-1 if i == j else 1 for j in range(ndim)])
        for i, v in enumerate(vec)
    ]
    return tuple(broadcasted)


def fourier_resample(
    image: torch.Tensor,
    scale_factor: int | float | Tuple[int, ...] | Tuple[float, ...],
):
    """Resample an N-dimensional image using Fourier transforms.

    This function performs image resampling (interpolation or binning) by
    manipulating the frequency components in Fourier space. For upsampling
    (scale_factor > 1), the frequency spectrum is zero-padded. For downsampling
    (scale_factor < 1), high frequency components are removed.

    Args:
        image: Input tensor of shape (..., H, W) to be resampled.
        scale_factor: Factor by which to scale the image. Can be either:
            - A single number: The same factor is applied to all dimensions
            - A tuple: Different scaling factors for each dimension.
              Must match the number of dimensions in the input image.
              Values > 1 will upsample, values < 1 will downsample.

    Returns:
        torch.Tensor: Resampled image with scaled dimensions. If the input shape
        is (D1, D2, ..., Dn) and scale factors are (s1, s2, ..., sn), the output
        shape will be (round(s1*D1), round(s2*D2), ..., round(sn*Dn)).

    Examples:
        # Uniform 2x upsampling in all dimensions
        >>> img = torch.randn(100, 100)
        >>> result = fourier_resample(img, 2.0)  # Shape: (200, 200)

        # Different scaling per dimension
        >>> img = torch.randn(100, 100, 100)
        # Shape: (200, 50, 100)
        >>> result = fourier_resample(img, (2.0, 0.5, 1.0))

    Notes:
        - The function uses real FFT (rfftn) for efficiency, as input is
          assumed to be real.
        - The implementation preserves the frequency content of the original
          image within the Nyquist limits of the target resolution.
        - For exact integer scaling factors, this method is equivalent to ideal
          interpolation or binning.
    """
    orig_shape = image.shape

    if isinstance(scale_factor, (int, float)):
        scale_factors = (scale_factor,) * image.ndim
    elif isinstance(scale_factor, tuple):
        assert (
            len(scale_factor) == image.ndim
        ), "scale_factors must match image dimensionality"
        scale_factors = scale_factor

    scaled_shape = tuple(int(f * s) for f, s in zip(scale_factors, orig_shape))

    # take the FT of input image
    image_ft = fft.rfftn(image)

    # Get the shape of the frequency domain after scaling
    scaled_ft_shape = compute_rfftn_size(*scaled_shape)

    # Initialize output array
    scaled_ft = torch.zeros(scaled_ft_shape, dtype=torch.complex64)

    # For each axis, determine whether we're upsampling or downsampling
    # and compute the appropriate slices
    slices = []
    for i, (orig_size, scaled_size, factor) in enumerate(
        zip(orig_shape, scaled_shape, scale_factors)
    ):
        if i == len(orig_shape) - 1:  # Last dimension (real FFT)
            orig_freq = freq_index(orig_size, half=True)
            scaled_freq = freq_index(scaled_size, half=True)
        else:
            orig_freq = freq_index(orig_size, half=False)
            scaled_freq = freq_index(scaled_size, half=False)

        if factor >= 1:
            # Upsampling: take all original frequencies
            slice_size = len(orig_freq)
            slices.append(slice(0, slice_size))
        else:
            # Downsampling: take subset of frequencies
            slice_size = len(scaled_freq)
            slices.append(slice(0, slice_size))

    # Create views into the original and scaled arrays using computed slices
    if all(f >= 1 for f in scale_factors):
        # All upsampling: copy original into subset of new array
        scaled_ft[tuple(slices)] = image_ft
    else:
        # At least some downsampling: copy subset of original into new array
        scaled_ft = image_ft[tuple(slices)]

    # compute intensity normalization factor
    Norig = torch.prod(torch.tensor(orig_shape))
    Nscaled = torch.prod(torch.tensor(scaled_shape))
    norm_factor = Nscaled / Norig

    # do inverse transform
    return fft.irfftn(scaled_ft * norm_factor, s=scaled_shape)
