from typing import Tuple, Union

import numpy as np
import scipy.fft as fft

# 5-tap coefficients
indices = np.array([0, 1, 2, -1, -2])
coefs1 = np.array([0, 8, -1, 1, -8]) / 12.0
coefs2 = np.array([-30, 16, -1, -1, 16]) / 12.0


def compute_first_derivative_filters(ndim: int):
    shape = [5] * ndim
    kernels = []

    for d in range(ndim):
        kernel = np.zeros(shape, dtype=np.float32)
        idx_shape = [1] * ndim
        idx_shape[d] = -1
        np.put_along_axis(
            kernel,
            indices.reshape(idx_shape),
            coefs1.reshape(idx_shape),
            axis=d,
        )
        kernels.append(kernel)

    return kernels


def compute_second_derivative_filters(ndim: int):
    """compute real-space kernel for second derivatives

    Args:
    ndim (int): number of dimensions

    Returns:
    np.ndarray containing real-space kernels
    """

    coords = [indices for _ in range(ndim)]
    grids = np.meshgrid(*coords, indexing="ij")

    kernels = [coefs2[grid].astype(np.float32) for grid in grids]

    # Mixed derivatives (off-diagonal terms)
    for i in range(ndim):
        for j in range(i + 1, ndim):
            kernels.append(
                coefs1[grids[i]].astype(np.float32)
                * coefs1[grids[j]].astype(np.float32)
            )

    return kernels


def fourier_meshgrid(
    shape: Tuple[int], spacing: Union[int, Tuple[int]]
) -> np.ndarray:
    """compute frequency spacing as meshgrid for n-dimensional arrays.
    The output coordinates assume real signals for memory efficiency.
    """
    ndim = len(shape)

    if isinstance(spacing, (int, float)):
        spacing = [
            spacing,
        ] * ndim
    else:
        assert (
            len(spacing) == ndim
        ), "number of spacing must equal to the number of shape."

    freqs = [
        fft.fftfreq(n, d=Δ) if i < (ndim - 1) else fft.rfftfreq(n, d=Δ)
        for i, (n, Δ) in enumerate(zip(shape, spacing))
    ]

    fgrids = np.meshgrid(*freqs, indexing="ij")

    return fgrids


def compute_3D_second_order_derivative_filters(
    shape: Tuple[int],
    spacing: Union[int, Tuple[int]] = 1,
    lateral_to_axial_ratio=1.0,
) -> np.ndarray:
    """
    The first axis is assumed to be the Z-axis (axial direction)

    Returns (6,) + shape complex ndarray
    """

    assert len(shape) == 3, "shape must be for a 3D image"

    fgrids = fourier_meshgrid(shape, spacing)

    twopi_i = 2.0 * np.pi * 1j
    sqrt2 = np.sqrt(2.0)
    δ = lateral_to_axial_ratio
    δ2 = δ * δ

    # compute diagonal terms, (e.g. ∂_zz, ∂_yy, ∂_xx, ...)
    filters = []

    for i, k in enumerate(fgrids):
        if i == 0:
            f = δ2 * sum(
                [
                    c * np.exp(-twopi_i * ci * k)
                    for c, ci in zip(coefs2, indices)
                ]
            )
        else:
            f = sum(
                [
                    c * np.exp(-twopi_i * ci * k)
                    for c, ci in zip(coefs2, indices)
                ]
            )
        filters.append(f)

    # compute mixed directional filters (e.g. ∂_yz, ∂_xz, ∂_yx)
    # only for the upper-diagonal parts due to symmetry, ∂_yx = ∂_xy
    # compute mixed directional filters (∂_yz, ∂_xz, ∂_xy)
    for i, ki in enumerate(fgrids):
        for j, kj in enumerate(fgrids[i + 1 :], start=i + 1):
            if i == 0:  # If one of the dimensions is z (axial)
                f = (
                    sqrt2
                    * δ
                    * sum(
                        [
                            c * np.exp(-twopi_i * ci * ki)
                            for c, ci in zip(coefs1, indices)
                        ]
                    )
                    * sum(
                        [
                            c * np.exp(-twopi_i * ci * kj)
                            for c, ci in zip(coefs1, indices)
                        ]
                    )
                )
            else:  # For lateral dimensions (x-y plane)
                f = (
                    sqrt2
                    * sum(
                        [
                            c * np.exp(-twopi_i * ci * ki)
                            for c, ci in zip(coefs1, indices)
                        ]
                    )
                    * sum(
                        [
                            c * np.exp(-twopi_i * ci * kj)
                            for c, ci in zip(coefs1, indices)
                        ]
                    )
                )
            filters.append(f)

    # return filters
    return np.stack(filters)
