import math
from dataclasses import dataclass

import numpy as np
import scipy.fft as fft
from scipy.special import j1


@dataclass
class MicroscopeParameters:
    """Parameters describing the microscope system"""

    excitation_wavelength: float  # in micrometers
    emission_wavelength: float  # in micrometers
    numerical_aperture: float
    sample_refractive_index: float
    immersion_refractive_index: float
    pixel_size: float  # in micrometers


def psf3d(
    Nz: int,
    Ny: int,
    Nx: int,
    dxy: float,
    dz: float,
    wvlen: float = 0.530,
    NA: float = 1.4,
    nobj: float = 1.4,
    noil: float = 1.515,
    N: int = 1024,
    t: float = 0.0,
):
    """compute 3D PSF using a simplified (Gibson & Lanni) 2-layer model

    coverslip glass thickness layer is omitted. To mitigate boundary effect
    from FFTs, PSFs should be computed at larger N before cropping it to the
    desired / requested size (Nx and Ny). The 3D PSF can be computed with
    different depth by varying the parameter `t`. This number sh

    Args:
    -----
    Nz : number of z slices
    Ny : psf height
    Nx : psf width
    dxy: lateral pixel spacing, micron
    dz : axial pixel spacing, micron
    NA : numerical aperture
    nobj: refractive index of object
    noil: refractive index of immersion
    N: number of samples used for computing oversampled PSF
    t: distance of sample from coverslip, micron. Larger number means deeper
       into the sample.

    """
    # compute frequency grids in wavelength units
    max_size = max(Ny, Nx)

    assert (
        N > max_size
    ), f"Number of samples must be larger than requested Ny or Nx"

    assert t >= 0, "distance of sample from coverslip can't be negative"

    fy = fft.fftfreq(N, d=dxy / wvlen)
    fx = fft.fftfreq(N, d=dxy / wvlen)
    Ky, Kx = np.meshgrid(fy, fx, indexing="ij")

    # ideal complex pupil
    # band limit is at NA/wvlen on pupil plane
    # on the OTF, the bandlimit is at 2 * NA/wvlen
    R = np.sqrt(Kx**2 + Ky**2)
    pupil = (R < NA) + 0j

    # compute the angles of rays from object
    sin_theta_obj = np.clip(R / nobj, -1, 1)
    sin_theta_oil = np.clip(R / noil, -1, 1)

    theta_obj = np.arcsin(sin_theta_obj)
    theta_oil = np.arcsin(sin_theta_oil)

    # compute optical path lengths along optical axis
    # sample depth from coverslip (t_s in G&L)
    op_obj = t * nobj * np.cos(theta_obj)

    # compute z-indices (use fft even/odd sample convention)
    zi = np.r_[: Nz // 2 + (Nz % 2 != 0), -Nz // 2 : 0]

    # compute z planes in physical units (micron) & account for RI mismatch
    zplanes = dz * (nobj / noil) * zi

    # compute optical path lengths for all defocus planes
    # these are t_i (thickness of 'immersion' in G&L)
    # this is actually t_i - t_i* (relative difference to design distance)
    # so when z = 0, spherical aberration only comes from `t_s` path length
    op_oil = zplanes[:, None, None] * noil * np.cos(theta_oil)

    # compute optical path difference: actual - design
    opd = op_obj[None, :, :] - op_oil

    # apply phase aberration to pupil
    pupil = pupil * np.exp(2 * np.pi * 1j * opd)

    apsf = fft.ifft2(pupil, axes=(-1, -2))

    psf = np.abs(apsf) ** 2
    psf /= psf.max()

    # crop PSF
    yind = np.r_[: Ny // 2 + (Ny % 2 != 0), -Ny // 2 : 0]
    xind = np.r_[: Nx // 2 + (Nx % 2 != 0), -Nx // 2 : 0]
    yi, xi = np.ix_(yind, xind)

    return psf[:, yi, xi]


def jinc(Nx: int, Ny: int, dxy: float, wavelength=0.530, NA=1.4):
    """calculate 2D jinc function (Airy pattern) for 2D PSF

    This is a real-space computation.

    Adapted from PSFToolbox-MATLAB from 'bionanoimaging'
    https://github.com/bionanoimaging/PSFToolbox-Matlab/

    For comparison, the pupil-space coordinates are computed as such:
    kmax = NA / λ (inverse micron unit)
    fy, fx = fftfreq(Ny, d=dy), fftfreq(Nx, d=dx)
    ky, kx = meshgrid(fy, fx, indexing="ij")
    k = sqrt(kx**2 + ky**2)
    pupil = k <= kmax

    """

    # create a radial distance
    scale = NA / (wavelength / dxy)
    ry = np.fft.fftfreq(Ny) * Ny * scale
    rx = np.fft.fftfreq(Nx) * Nx * scale
    Y, X = np.meshgrid(ry, rx, indexing="ij")
    R = np.sqrt(X**2 + Y**2)
    nonzero = R != 0
    res = np.zeros_like(R)
    res[nonzero] = j1(2 * np.pi * R[nonzero]) / (np.pi * R[nonzero])
    res[~nonzero] = 1.0

    return res


def Zernike_polynomial(rho, phi, nmax):
    """computes zernike polynomials

    Args:
    rho (np.ndarray): normalized radial coordinate, [0, 1].
    phi (np.ndarray): angular coordinates. Typically arctan2(k_y /k_x)
    nmax (int): number of polynomials to be computed, minus one

    Returns:
    np.ndarray, circular zernike polynomials
    """

    def _Rmn(m, n, rho):
        Rmn = np.zeros_like(rho)
        Nk = int((n - m) / 2)
        for k in range(Nk + 1):
            _num = (-1.0) ** k * math.factorial(n - k)
            try:
                _den = (
                    math.factorial(k)
                    * math.factorial(int((n + m) / 2 - k))
                    * math.factorial(int((n - m) / 2 - k))
                )
            except ValueError:
                print("m={:d}, n={:d}, k={:d}".format(m, n, k))
                raise
            Rmn += (float(_num) / float(_den)) * rho ** (float(n - 2 * k))
        return Rmn

    def _recursive_sum(n):
        if n == 0:
            return 0
        else:
            return n + _recursive_sum(n - 1)

    Norders = _recursive_sum(nmax + 1)
    Zmn = np.zeros((Norders,) + rho.shape, dtype=float)

    for n in range(nmax, -1, -1):
        for m in range(-n, n + 2, 2):
            n = int(n)
            m = int(m)
            normfactor = (float(2 * (n + 1)) / (1.0 + float(m == 0))) ** 0.5
            # ANSI notation
            j = int((n * (n + 2) + m) / 2)
            if m >= 0:
                # even function
                Zmn[j, :, :] = (
                    normfactor * _Rmn(m, n, rho) * np.cos(float(m) * phi)
                )
            elif m < 0:
                # odd function
                Zmn[j, :, :] = (
                    -normfactor * _Rmn(-m, n, rho) * np.sin(-float(m) * phi)
                )

    return Zmn
