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


class PupilFunction:
    """Class to handle pupil function calculations and manipulations"""

    def __init__(
        self,
        params: MicroscopeParameters,
        Nx: int = 128,
        Ny: int = 128,
        oversampling=2,
    ):
        """
        Initialize pupil function calculator

        For zernike polynomial calculations, the quantities for the
        radial coordinate ρ and angular coordinate ϕ are obtained from

        ρ = PupilFunction.R / (NA/λ_em) # only compute where ρ ∈ [0, 1]
        ϕ = PupilFunction.Phi

        Also, you can resample the pupil for computing PSFs with different
        wavelength by scaling ρ × (λ₁ / λ₀) where λ₀ is the wavelength used
        for the phase retrieval procedure

        Args:
            params: MicroscopeParameters object
            size: Size of pupil function array (default 512x512)
        """
        self.params = params
        self.Ny = Ny
        self.Nx = Nx
        self._initialize_coordinates()

    def _initialize_coordinates(self, use_excitation_wavelength=False):
        """Initialize coordinate systems for calculations"""
        x = fft.fftfreq(self.Nx, d=self.params.pixel_size)
        y = fft.fftfreq(self.Ny, d=self.params.pixel_size)
        self.Y, self.X = np.meshgrid(y, x, indexing="ij")
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Phi = np.arctan2(self.Y, self.X)

        if use_excitation_wavelength:
            wavelength = self.params.excitation_wavelength
        else:
            wavelength = self.params.emission_wavelength

        # Create pupil mask based on NA
        bandlimit = self.params.numerical_aperture / wavelength
        self.mask = self.R <= bandlimit + 0j

        # compute compression factor 1/sqrt(cos(θ₁))
        # on the objective lens side
        sin_theta1 = np.clip(
            wavelength / self.params.immersion_refractive_index * self.R,
            -1,
            1,
        )
        sin_theta2 = np.clip(
            wavelength / self.params.sample_refractive_index * self.R,
            -1,
            1,
        )
        #          n_s           ||       n_i
        #  (source) ∠θ₂ ----> refraction --> ∠θ₁
        # angles of rays of light leaving the immersion, refracted from
        # the sample upon 'seeing' immersion boundary
        self.theta_1 = np.arcsin(sin_theta1)

        # angles of rays of light from source (sample)
        self.theta_2 = np.arcsin(sin_theta2)

    def calculate_ideal_pupil(
        self, use_excitation_wavelength=False
    ) -> np.ndarray:
        """Calculate ideal pupil function without aberrations"""

        if use_excitation_wavelength:
            wavelength = self.params.excitation_wavelength
        else:
            wavelength = self.params.emission_wavelength

        airy = jinc(
            self.Nx,
            self.Ny,
            self.params.pixel_size,
            wavelength,
            self.params.numerical_aperture,
        )

        pupil = fft.fft2(airy)

        pupil /= np.sqrt(np.cos(self.theta_1))

        # clip outside of pupil
        pupil *= self.mask

        return pupil

    def calculate_3d_psf(
        self,
        z_positions: np.ndarray,
        zpos: float = 0.0,
        use_excitation_wavelength=False,
        return_amplitude_psf=False,
    ) -> np.ndarray:
        """
        Calculate 3D (widefield) PSF from pupil function

        Args:
            z_positions: Array of z positions to calculate PSF at

        Returns:
            3D array containing PSF
        """

        if use_excitation_wavelength:
            self._initialize_coordinates(use_excitation_wavelength=True)
            wavelength = self.params.excitation_wavelength
        else:
            wavelength = self.params.emission_wavelength

        k0sq = (self.params.immersion_refractive_index / wavelength) ** 2

        krsq = self.R**2
        kz = np.sqrt(k0sq - krsq + 0j)

        # Add z dimension for broadcasting
        kz = kz[np.newaxis, :, :]
        z_positions = z_positions[:, np.newaxis, np.newaxis]

        # Calculate defocus phase for all z positions at once
        phase = np.exp(1j * 2 * np.pi * kz * z_positions)
        pupil = self.calculate_ideal_pupil(
            use_excitation_wavelength=use_excitation_wavelength
        )

        # calculate phase from z position
        OPd = zpos * (
            self.params.sample_refractive_index * np.cos(self.theta_2)
            - self.params.immersion_refractive_index * np.cos(self.theta_1)
        )

        zphase = np.exp(1j * 2 * np.pi * OPd / wavelength)

        # Apply pupil and defocus
        defocused_pupils = (
            pupil[np.newaxis, :, :]
            * phase
            * zphase[np.newaxis, :, :]
            * self.mask
        )

        psf_amplitude = fft.ifft2(defocused_pupils, axes=(-2, -1))

        if return_amplitude_psf:
            return psf_amplitude
        else:
            psfi = np.abs(psf_amplitude) ** 2
            psfi /= psfi.max()
            return psfi

    def calculate_confocal_3D_psf(
        self,
        z_positions: np.ndarray,
        zpos: float = 0.0,
    ):
        # calculate emission amplitude PSF
        psf_em = self.calculate_3d_psf(
            z_positions,
            zpos=zpos,
            use_excitation_wavelength=False,
            return_amplitude_psf=True,
        )

        # because of different band limit, recalculate pupil coordinates
        self._initialize_coordinates(use_excitation_wavelength=True)
        psf_ex = self.calculate_3d_psf(
            z_positions,
            zpos=zpos,
            use_excitation_wavelength=True,
            return_amplitude_psf=True,
        )

        # reset coordinates just in case
        self._initialize_coordinates()

        confocal_psf = np.abs(psf_ex * psf_em.conj()) ** 2
        confocal_psf /= confocal_psf.max()

        return confocal_psf


def compute_z_planes(Nz: int, dz: float):
    # add offset of 1 if odd
    if Nz % 2 == 0:
        w = int(Nz / 2)
    else:
        w = int((Nz - 1) / 2)

    return np.r_[: w + (Nz % 2 == 0), -w:0] * dz


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
