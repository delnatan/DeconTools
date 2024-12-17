import numpy as np
import scipy.fft as fft


def perfft(img: np.ndarray, inverse: bool = True) -> np.ndarray:
    """Compute Moisan's Periodic plus Smooth Image Decomposition.

    This function decomposes an image into periodic and smooth components using
    Moisan's algorithm. It handles n-dimensional inputs (typically 2D or 3D
    images).

    Parameters
    ----------
    img : np.ndarray
        Input image array of any dimension (typically 2D or 3D).
        Will be converted to float64 if not already.
    inverse : bool, optional
        If True, returns the periodic component in spatial domain (default).
        If False, returns the periodic component in frequency domain.

    Returns
    -------
    np.ndarray
        If inverse=True: Periodic component in spatial domain
        If inverse=False: Fourier transform of periodic component

    Notes
    -----
    The algorithm computes a boundary image where discontinuities across
    periodic boundaries are captured. For each dimension d:
        s[first_d] = img[last_d] - img[first_d]
        s[last_d] = -(s[first_d])

    References
    ----------
    L. Moisan, "Periodic plus Smooth Image Decomposition",
    Journal of Mathematical Imaging and Vision, vol 39:2, pp. 161-179, 2011.

    """
    img = img.astype(np.float64)
    dims = img.shape
    ndim = img.ndim

    # boundary image, 'smooth' part
    s = np.zeros_like(img)

    for dim in range(ndim):
        # each slice(None) creates slice(None, None, None)
        first_idx = [slice(None)] * ndim
        last_idx = [slice(None)] * ndim
        first_idx[dim] = 0
        last_idx[dim] = -1
        first_idx = tuple(first_idx)
        last_idx = tuple(last_idx)

        boundary_diff = img[last_idx] - img[first_idx]

        if dim == 0:
            s[first_idx] = boundary_diff
            s[last_idx] = -boundary_diff
        else:
            s[first_idx] += boundary_diff
            s[last_idx] -= boundary_diff

    freq_coords = [2 * np.pi * fft.fftfreq(s) for s in img.shape]

    denom = np.zeros_like(img)
    # compute laplace solution in frequency space
    # ndim * (∑(cos(2π*k/N)) - 2.0)
    for dim, freq in enumerate(freq_coords):
        shape = [1] * img.ndim
        shape[dim] = len(freq)
        freq = freq.reshape(shape)
        denom += np.cos(freq)

    denom -= 2.0
    denom *= 2 * img.ndim

    origin_idx = tuple(0 for _ in range(img.ndim))

    denom[origin_idx] = 1.0
    S = fft.fftn(s) / np.maximum(denom, 1e-10)
    S[origin_idx] = 0.0

    P = fft.fftn(img) - S

    if inverse:
        return fft.ifftn(P)
    else:
        return P
