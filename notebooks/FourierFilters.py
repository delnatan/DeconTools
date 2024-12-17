import numpy as np


# demo function for prototyping
def compute_filters(Nx):
    """
    Finite difference operator as Fourier filters

    Using Taylor expansion with stencils. Usually this kind of accuracy is not
    needed in deconvolution problems.

    See
    - Abramowitz & Stegun 'Finite differences':
    - https://web.media.mit.edu/~crtaylor/calculator.html

    """
    stencil = np.array([-2, -1, 0, 1, 2], dtype=float)
    coefs = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0
    coefs2 = np.array([-1, 16, -30, 16, -1], dtype=float) / 12.0
    twopi = 2.0 * np.pi

    # use Fourier shift theorem to compute filters

    # fourier coordinates
    kx = np.fft.rfftfreq(Nx)

    # fourier filters
    Lx = np.zeros(kx.size, dtype=np.complex128)
    Lxx = np.zeros(kx.size, dtype=np.complex128)

    # each stencil element is the offset position
    for i, s in enumerate(reversed(stencil)):
        Lx += coefs[i] * np.exp(-twopi * 1j * kx * s)
        Lxx += coefs2[i] * np.exp(-twopi * 1j * kx * s)

    return Lx, Lxx


def fourier_meshgrid(*N, d=1.0):
    """computes meshgrid of frequency spacing

    Note: only half coordinates are considered (for real-valued input)

    Args:
        N1,N2,N3,... : int
            Size of sample N for computing Fourier coordinate grids
        d1,d2,d3... : float, optional
            Real-space spacing for each corresponding sample size N1,N2,...
            default is 1.0

    """

    Ndim = len(N)
    dimrange = range(Ndim)

    if d == 1.0:
        dvec = [1.0 for n in dimrange]

    else:
        dvec = d

    Nd = len(dvec)

    assert (
        Ndim == Nd
    ), "Number of spacing d ({:d}) must be equal to number of \
    sample sizes ({:d})".format(
        Nd, Ndim
    )

    # last dimension uses rfft by convention (so will have half the samples)

    spacings = []
    lastdim = dimrange[-1]

    for n in dimrange:
        if n == lastdim:
            spacings.append(np.fft.rfftfreq(N[n], d=dvec[n]))
        else:
            spacings.append(np.fft.fftfreq(N[n], d=dvec[n]))

    return np.meshgrid(*spacings, indexing="ij")


def second_order_diffops_2d(Ny, Nx):
    ky, kx = fourier_meshgrid(Ny, Nx)
    twopi_i = 2.0 * np.pi * 1j
    sqrt2 = np.sqrt(2.0)
    Lyy = np.exp(-twopi_i * ky) - 2.0 + np.exp(twopi_i * ky)
    Lxx = np.exp(-twopi_i * kx) - 2.0 + np.exp(twopi_i * kx)
    Lxy = sqrt2 * (
        1
        - np.exp(-twopi_i * kx)
        - np.exp(-twopi_i * ky)
        + np.exp(-twopi_i * (kx + ky))
    )
    return Lyy, Lxx, Lxy


def second_order_diffops_3d(Nz, Ny, Nx, lateral_to_axial_ratio=1.0):
    """second-order finite difference as Fourier filters

    Taken from Arigovindan et al. 2013 (supplemental info)

    Args:
        Nz (int):
            number of axial slices
        Ny (int):
            number of rows
        Nx (int):
            number of columns
        lateral_to_axial_ratio (float):
            dz/dx in pixel spacing to scale the finite-difference spacing

    Returns:
        Fourier filters in zz, yy, xx, xy, yz, xz

    """
    kz, ky, kx = fourier_meshgrid(Nz, Ny, Nx)

    twopi_i = 2.0 * np.pi * 1j
    sqrt2 = np.sqrt(2.0)
    δ = lateral_to_axial_ratio

    Lzz = δ ** 2 * (np.exp(-twopi_i * kz) - 2.0 + np.exp(twopi_i * kz))
    Lyy = np.exp(-twopi_i * ky) - 2.0 + np.exp(twopi_i * ky)
    Lxx = np.exp(-twopi_i * kx) - 2.0 + np.exp(twopi_i * kx)
    Lxy = sqrt2 * (
        1
        - np.exp(-twopi_i * kx)
        - np.exp(-twopi_i * ky)
        + np.exp(-twopi_i * (kx + ky))
    )
    Lyz = (
        sqrt2
        * δ
        * (
            1
            - np.exp(-twopi_i * ky)
            - np.exp(-twopi_i * kz)
            + np.exp(-twopi_i * (ky + kz))
        )
    )
    Lxz = (
        sqrt2
        * δ
        * (
            1
            - np.exp(-twopi_i * kx)
            - np.exp(-twopi_i * kz)
            + np.exp(-twopi_i * (kx + kz))
        )
    )

    return Lzz, Lyy, Lxx, Lxy, Lyz, Lxz


def atrous_filters(shape, levels=4):
    """compute discrete wavelet transform (Fourier) filters

    Note that the last filter is the for computing last smoothed image,
    required for reconstruction. Coefficients use cubic b-splines.

    F^(0) = F^(k_max) + sum(W^(k))

    k_max is the number of decompisition level + 1
    W are the wavelet coefficients
    F^(k_max) is the last blurred image

    Args:
        shape : list of ints
            Each integer is the shape of input dimensions

    Returns:
        Fourier filters with the shape (levels, dim[0], dim[1], ...)

    Example ::
        img = imread("roi_0014.png")
        x̃ = np.fft.rfft2(img.astype(float))
        ψ = atrous_filters(img.shape, levels=4)
        # use broadcasting to do the convolution
        coefs = np.fft.irfft2(ψ * x̃[None, :, :])

        # reconstruction is done by summing the coefficients
        recon = coefs.sum(axis=0)

    """
    kspace = fourier_meshgrid(*shape)

    fourier_shape = kspace[0].shape
    psi = np.zeros((levels + 1, *fourier_shape), dtype=np.complex128)
    twopi_i = 2.0 * np.pi * 1j

    wrk = np.ones(fourier_shape, dtype=np.complex128)
    for s in kspace:
        wrk *= (
            0.0625 * np.exp(-twopi_i * -2 * s)
            + 0.25 * np.exp(-twopi_i * -1 * s)
            + 0.375
            + 0.25 * np.exp(-twopi_i * 1 * s)
            + 0.0625 * np.exp(-twopi_i * 2 * s)
        )

    psi[0, ...] = wrk.copy()

    # compute the "holey" spline filters
    for k in range(levels):
        wrk = np.ones(fourier_shape, dtype=np.complex128)
        for s in kspace:
            wrk *= (
                0.0625 * np.exp(-twopi_i * -(2 ** k * 2 + 2) * s)
                + 0.25 * np.exp(-twopi_i * -(2 ** k + 1) * s)
                + 0.375
                + 0.25 * np.exp(-twopi_i * (2 ** k + 1) * s)
                + 0.0625 * np.exp(-twopi_i * (2 ** k * 2 + 2) * s)
            )
        psi[k + 1, ...] = wrk.copy()

    # form the wavelet filters
    WF = np.zeros((levels + 1, *fourier_shape), dtype=np.complex128)

    for k in range(levels):
        if k == 0:
            WF[k, ...] = 1.0 - psi[k]
        else:
            WF[k, ...] = psi[0:k, ...].prod(axis=0) - psi[0 : k + 1, ...].prod(
                axis=0
            )

    # last filter is a product of all coefficients (except the last one)
    WF[-1, ...] = psi[:-1].prod(axis=0)

    return WF


def butterworth_filter_2d(
    Ny, Nx, pixel_size, NA=1.40, wavelength=0.525, order=2
):
    """compute Butterworth frequency filter

    TODO: set a number ϵ ∈ (0,1) to set ϵ as fraction of amplitude allowed
    at frequency cutoff. solve for x after setting y * ϵ, then normalize ω by
    freq_cutoff/x.

    """

    fy = np.fft.fftfreq(Ny, d=pixel_size)
    fx = np.fft.rfftfreq(Nx, d=pixel_size)
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    # nyquist is at 1/(2*pixel_size); edge of frequency-space
    freq_cutoff = NA / wavelength
    # normalize ω so that at freq_cutoff, the value is 1
    ω = np.sqrt(ky ** 2 + kx ** 2) / freq_cutoff
    return 1 / np.sqrt(1.0 + ω ** (2 * order))


def butterworth_filter_3d(
    Nz,
    Ny,
    Nx,
    dxy,
    dz,
    NA=1.40,
    wavelength=0.525,
    order=2,
    f_lateral=1.0,
    f_axial=1.0,
):
    """compute Butterworth frequency filter

    TODO: set a number ϵ ∈ (0,1) to set ϵ as fraction of amplitude allowed
    at frequency cutoff. solve for x after setting y * ϵ, then normalize ω by
    freq_cutoff/x.

    Args:
        Nz (int): number of z-slices
        Ny (int): number of rows
        Nx (int): number of columns
        dxy (float): pixel spacing in xy (lateral)
        dz (float): pixel spacing in z (axial)
        order (int): order parameter for Butterworth filter, controls how fast
            high-frequency information decays, higher-order means faster decay
        f_lateral (float): fudge factor for lateral resolution cutoff. Higher
            means is more severe, default is 1.0
        f_axial (float): fudge factor for axial resolution cutoff.
    """
    fz = np.fft.fftfreq(Nz, d=dz)
    fy = np.fft.fftfreq(Ny, d=dxy)
    fx = np.fft.rfftfreq(Nx, d=dxy)
    kz, ky, kx = np.meshgrid(fz, fy, fx, indexing="ij")
    # lateral resolution cut-off is NA/λ
    kxy = (ky ** 2 + kx ** 2) / ((NA / wavelength) / f_lateral) ** 2
    # axial resolution cutoff is NA^2/(4λ)
    kz_ = (kz ** 2) / ((NA ** 2 / (4 * wavelength)) / f_axial) ** 2
    ω = np.sqrt(kz_ + kxy)
    return 1 / np.sqrt(1.0 + ω ** (2 * order))
