"""
PyTorch implementation of Bridget Hanser's pupil function

Note that 'd' here, referring to the distance from coverslip actually
only accounts for the spherical aberration (deviation from optimal condition),
it does NOT physically displace the PSF by 'd' microns.

If you actually want to move the PSF, this can be easily done by adding
a positive shift to the input vector z, the axial position where each slice
of the PSF is computed.

History:

D.E. Feb. 11 2025, revised spherical aberration calculation
                   added ZernikePolynomials

Current workflow for phase retrieval:
1) Record beads
2) Distill beads using recipes.distill_PSF
3) Retrieve pupil function
4) Calculate empirical OTF scaling
5) Fit to Zernike polynomials
6) Save Zernike coefficients and OTF scaling factor to file

To compute PSF at arbitrary resolution at-will:
7) Load zernike coefficients and scaling factor
8) Instantiate and compute pupil for a given wavelength from zernike coefficients
9) Compute PSF at arbitrary resolution

"""

import logging
import math
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import curve_fit

from ..utils.reduce import cylindrical_average_otf

EPS = torch.finfo(torch.float32).eps

logger = logging.getLogger("DeconTools")


def calculate_frequency_indices(n) -> torch.Tensor:
    positive_indices = torch.arange(0, n // 2 + (n % 2 != 0))
    negative_indices = torch.arange(-(n - (n % 2 != 0)) // 2, 0)
    return torch.cat([positive_indices, negative_indices])


def calculate_sphere_form_factor(
    Nz: int, Ny: int, Nx: int, dxy: float, dz: float, radius: float
):
    """computes normalized spherical form factor

               sin(kR) - kR cos(kR)
    F(k) =  3 ---------------------
                      (kR)³

    This form is commonly used in small angle scattering.
    Normalized such that F(0) = 1.0

    """
    _2pi = 2.0 * torch.pi
    fz = _2pi * torch.fft.fftfreq(Nz, d=dz)
    fy = _2pi * torch.fft.fftfreq(Ny, d=dxy)
    fx = _2pi * torch.fft.rfftfreq(Nx, d=dxy)

    kz, ky, kx = torch.meshgrid(fz, fy, fx, indexing="ij")

    K = torch.sqrt(kz**2 + ky**2 + kx**2)

    eps = torch.finfo(torch.float32).eps

    F = torch.where(
        K > 0,
        3
        * (torch.sin(K * radius) - K * radius * torch.cos(K * radius))
        / (torch.pow(K * radius, 3) + eps),
        1.0,
    )

    return F


class PupilFunction:
    def __init__(
        self,
        Nx: int = 512,
        Ny: int = 512,
        NA: float = 1.40,
        ni: float = 1.515,
        ns: float = 1.33,
        dxy: float = 0.085,
        wavelength: float = 0.530,
    ):
        self.Nx: int = Nx
        self.Ny: int = Ny
        self.NA: float = NA
        self.ni: float = ni
        self.ns: float = ns
        self.dxy: float = dxy
        # because rays entering the objective can't have half-angle >90
        # sin(90°) = 1, NA = n_s * 1.0
        self.effNA: float = min(self.NA, self.ns)
        self.wavelength: float = wavelength
        self.pupil = None  # phase-retrieved pupil function
        self.otf_scale = None  # gaussian OTF scaler
        self.compute_pupil_coordinates()

    def compute_pupil_coordinates(self):
        # this is now handled by the 'jinc' calculation
        # k_max is the bandlimit of the pupil function
        # k_max = self.effNA / self.wavelength
        freq_y = torch.fft.fftfreq(self.Ny, d=self.dxy)
        freq_x = torch.fft.fftfreq(self.Nx, d=self.dxy)
        self.ky, self.kx = torch.meshgrid(freq_y, freq_x, indexing="ij")
        self.kr = torch.sqrt(self.kx**2 + self.ky**2)
        sin_theta_immersion = torch.clip(
            self.wavelength * self.kr / self.ni, -1, 1
        )

        self.theta_immersion = torch.arcsin(sin_theta_immersion)

        # for vectorial computation, need φ
        self.varphi = torch.arctan2(self.ky, self.kx)

        # compute pupil mask
        self.mask = self._jinc_aperture(self.wavelength)
        self.hard_mask = (self.kr <= (self.effNA / self.wavelength)) * 1.0

    def _jinc_aperture(
        self, wavelength: float = 0.530, apply_sigma=True, kmax=0.5
    ):
        nx = calculate_frequency_indices(self.Nx)
        ny = calculate_frequency_indices(self.Ny)
        jx = nx * self.effNA * self.dxy / wavelength
        jy = ny * self.effNA * self.dxy / wavelength
        jY, jX = torch.meshgrid(jy, jx, indexing="ij")
        jR = torch.sqrt(jX * jX + jY * jY)
        pi_jR = torch.pi * jR
        jinc = torch.special.bessel_j1(2 * pi_jR) / torch.where(
            pi_jR > EPS, pi_jR, EPS
        )

        # apply Duchon's sigma factor (Lanczos filter) to mitigate ringing
        if apply_sigma:
            cutoff = kmax * max(jx.max(), jy.max())
            arg = torch.pi * (jR / cutoff)
            sigma = torch.sin(arg + EPS) / (arg + EPS)
            jinc *= torch.pow(sigma, 2)

        mask = torch.fft.fft2(jinc).real
        mask /= mask.max()
        mask[mask < EPS] = 0.0

        return mask

    def calculate_axial_coordinate(self):
        return torch.sqrt(
            torch.clamp((self.ni / self.wavelength) ** 2 - self.kr**2, min=0.0)
        )

    def calculate_defocus_phase(self, z: float | torch.Tensor):
        kz = self.calculate_axial_coordinate()

        if isinstance(z, float):
            return torch.exp(2j * torch.pi * kz * z)
        else:
            # use broadcasting to compute defocus terms for all axial/focal
            # positions
            return torch.exp(2j * torch.pi * kz[None, :, :] * z[:, None, None])

    def calculate_spherical_aberration_factor(
        self, d: float = 0.0, ns: float = 1.33
    ):
        """compute pupil aberration due to refractive index mismatch"""
        sin_theta_sample = torch.clip(self.wavelength * self.kr / ns, -1, 1)
        theta_sample = torch.arcsin(sin_theta_sample)

        opd = (
            -d
            * (
                self.ns * torch.cos(theta_sample)
                - self.ni * torch.cos(self.theta_immersion)
            )
            / self.wavelength
        )

        # Wavefront compression factor A_w
        A_w = torch.where(
            self.theta_immersion < EPS,
            self.ni / self.ns,
            self.ni
            * torch.tan(theta_sample)
            / (self.ns * torch.tan(self.theta_immersion)),
        )

        # Fresnel factor
        sin_theta_sum = torch.sin(self.theta_immersion + theta_sample)
        cos_theta_diff = torch.cos(theta_sample - self.theta_immersion)

        A_t = (
            torch.sin(self.theta_immersion)
            * torch.cos(theta_sample)
            / (sin_theta_sum + EPS)
        ) * (1 + 1 / cos_theta_diff)

        # ψ / λ in Hanser's paper
        return A_t * A_w * torch.exp(2j * torch.pi * opd) / self.wavelength

    def calculate_3d_psf(
        self,
        Nz: int,
        dz: float,
        ns: float = 1.33,
        device: str | torch.device = "cpu",
        source_depth: float = 0.0,
        use_pupil: bool = False,
        scale_otf: bool = False,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        else:
            device = device

        z = calculate_frequency_indices(Nz) * dz

        if use_pupil:
            if self.pupil is None:
                raise AttributeError("Pupil has not been retrieved!")
            else:
                pupil = self.pupil.to(device)
        else:
            pupil = (self.mask + 0j).to(device)

        if ns != self.ni:
            psi = self.calculate_spherical_aberration_factor(
                source_depth, ns=ns
            )
            psi = psi.to(device)
            pupil *= psi

        axial_shift = self.calculate_defocus_phase(z).to(device)

        psfa = torch.fft.ifft2(pupil[None, :, :] * axial_shift)
        psfi = torch.abs(psfa) ** 2

        # scale OTF by 'fudge' factor
        if scale_otf:
            if self.otf_scale is not None:
                # compute gaussian roll-off along radial coordinate
                scale = torch.exp(-(torch.pi**2) * self.kr**2 / self.otf_scale)
                scale = scale.to(device)
                otf = torch.fft.fftn(psfi) * scale[None, :, :]
                psfi = torch.fft.ifftn(otf).magnitude
            else:
                logger.warn(
                    "scale_otf is True, but 'otf_scale' is None. OTF is not scaled."
                )

        psfi /= psfi.sum()

        return psfi

    def __repr__(self):
        class_name = self.__class__.__name__
        reprstr = ""
        reprstr += f"<{class_name}>\n"
        reprstr += f"Nx = {self.Nx}, Ny = {self.Ny}\n"
        reprstr += f"NA = {self.NA}, eff. NA = {self.effNA}\n"
        reprstr += f"ni = {self.ni}, ns = {self.ns}\n"
        reprstr += f"Δxy = {self.dxy} μm\n"
        reprstr += f"λ_em = {self.wavelength:.3f} μm\n"
        return reprstr


@lru_cache(maxsize=128)
def _factorial(n: int) -> int:
    """Cached factorial calculation"""
    if n == 0 or n == 1:
        return 1
    return n * _factorial(n - 1)


class ZernikePolynomials:
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device

    def _compute_radial_term(
        self, m: int, n: int, rho: torch.Tensor
    ) -> torch.Tensor:
        """Compute radial Zernike term R_n^m(rho)"""
        Rmn = torch.zeros_like(rho)
        Nk = (n - m) // 2

        for k in range(Nk + 1):
            num = (-1.0) ** k * _factorial(n - k)
            den = (
                _factorial(k)
                * _factorial((n + m) // 2 - k)
                * _factorial((n - m) // 2 - k)
            )
            Rmn += (num / den) * rho ** (n - 2 * k)

        return Rmn

    def compute_polynomials(
        self, rho: torch.Tensor, phi: torch.Tensor, nmax: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Zernike polynomials up to order nmax

        Args:
            rho: Normalized radial coordinates [0, 1], shape [H, W]
            phi: Angular coordinates, shape [H, W]
            nmax: Maximum radial order

        Returns:
            Tuple containing:
            - Zernike polynomials with shape [J, H, W] where J is number of polynomials
            - Indices array mapping (n,m) to polynomial index j
        """
        # Move inputs to correct device
        rho = rho.to(self.device)
        phi = phi.to(self.device)

        # Calculate total number of polynomials
        num_polynomials = ((nmax + 1) * (nmax + 2)) // 2

        # Initialize output array
        Zmn = torch.zeros((num_polynomials,) + rho.shape, device=self.device)

        # Store (n,m) to j index mapping
        indices = []

        # Generate polynomials
        for n in range(nmax + 1):
            for m in range(-n, n + 1, 2):
                # ANSI standard index
                j = (n * (n + 2) + m) // 2

                # Normalization factor
                norm = math.sqrt(2.0 * (n + 1) / (1.0 + float(m == 0)))

                indices.append((n, m, j))

                # Compute radial term
                Rmn = self._compute_radial_term(abs(m), n, rho)

                if m >= 0:
                    # Even function
                    Zmn[j] = norm * Rmn * torch.cos(m * phi)
                else:
                    # Odd function
                    Zmn[j] = -norm * Rmn * torch.sin(-m * phi)

        return Zmn, torch.tensor(indices, device=self.device)

    def fit_pupil(
        self,
        pupil: PupilFunction,
        nmax: int,
    ) -> dict[str, torch.Tensor]:
        """Fit pupil function to Zernike polynomials

        Args:
            pupil: PupilFunction
            nmax: Maximum radial order

        Returns:
            Zernike coefficients array
        """
        rho = pupil.kr.clone()
        phi = pupil.varphi.clone()
        kmax = pupil.effNA / pupil.wavelength

        # limit to unit circle
        rho = (rho / kmax) * pupil.hard_mask
        phi = phi * pupil.hard_mask

        # Compute polynomials
        Zmn, _ = self.compute_polynomials(rho, phi, nmax)

        # Reshape for leest squares
        pupil_mag = torch.abs(pupil.pupil)
        pupil_phase = torch.angle(pupil.pupil)

        Ymag = pupil_mag.reshape(-1)
        Yphase = pupil_phase.reshape(-1)

        X = Zmn.reshape(Zmn.shape[0], -1)

        # Solve least squares problem
        # mag_coeffs = torch.linalg.pinv(X.T) @ Ymag
        # phase_coeffs = torch.linalg.pinv(X.T) @ Yphase
        optmag = torch.linalg.lstsq(X.T, Ymag, rcond=None)
        optphase = torch.linalg.lstsq(X.T, Yphase, rcond=None)

        mag_coeffs = optmag.solution
        phase_coeffs = optphase.solution

        return {
            "magnitude": mag_coeffs,
            "phase": phase_coeffs,
        }

    def reconstruct_pupil(
        self,
        coeffs: dict[str, torch.Tensor],
        pupil: PupilFunction,
    ) -> torch.Tensor:
        """Reconstruct pupil from Zernike coefficients

        Args:
            coeffs: dictionary containing 'magnitude' and 'phase' coefficients
            pupil: PupilFunction
        """
        ncoeffs = len(coeffs["magnitude"])
        nmax = math.floor((-3 + math.sqrt(9 + 8 * ncoeffs)) / 2)

        rho = pupil.kr.clone()
        phi = pupil.varphi.clone()
        kmax = pupil.effNA / pupil.wavelength

        # limit to unit circle
        rho = (rho / kmax) * pupil.hard_mask
        phi = phi * pupil.hard_mask

        Zmn, _ = self.compute_polynomials(rho, phi, nmax)

        magnitude_coefs = coeffs["magnitude"]
        phase_coefs = coeffs["phase"]

        reconstructed_magnitude = torch.sum(
            magnitude_coefs[:, None, None] * Zmn, dim=0
        )
        reconstructed_phase = torch.sum(
            phase_coefs[:, None, None] * Zmn, dim=0
        )

        reconstructed_magnitude *= pupil.mask
        reconstructed_phase *= pupil.mask

        return reconstructed_magnitude * torch.exp(1j * reconstructed_phase)


class PhaseRetrievedPupilFunction:
    def __init__(
        self,
        data: torch.Tensor,
        dxy: float,
        dz: float,
        wavelength: float,
        NA: float,
        ni: float,
        ns: float,
    ):
        """Hanser's algorithm for phase-retrieved pupil function"""
        Nz, Ny, Nx = data.shape
        self.pf = PupilFunction(
            Nx=Nx, Ny=Ny, NA=NA, ni=ni, ns=ns, dxy=dxy, wavelength=wavelength
        )
        self.data = data
        self.dz = dz
        self.Nz = Nz

        # for debugging OTF scale fitting
        self._radial_otf_ratio = None
        self._r_freq = None
        self._debug = {}

    def apply_bead_size_correction(self, bead_diameter=0.200):
        B = calculate_sphere_form_factor(
            self.Nz,
            self.pf.Ny,
            self.pf.Nx,
            self.pf.dxy,
            self.dz,
            radius=bead_diameter / 2.0,
        )

        otf = torch.fft.rfftn(self.data)
        psf = torch.fft.irfftn(otf / B, s=self.data.shape)

        return torch.clamp(psf, min=0.0)

    def retrieve_phase(
        self,
        num_iters=50,
        zmin=-3.0,
        zmax=3.0,
    ):
        # calculate focal positions

        input_focal_planes = calculate_frequency_indices(self.Nz) * self.dz

        # get indices for a subset of the focal planes
        # zmin <= z <= zmax
        zidx = torch.argwhere(
            (zmin <= input_focal_planes) & (input_focal_planes <= zmax)
        ).squeeze()

        zpos = input_focal_planes[zidx]

        # apply bead size correction to measured psf
        psf = self.data
        # observed magnitudes = sqrt(intensity)
        obsmag = torch.sqrt(psf[zidx])

        # initial pupil
        pupil = self.pf.mask + 0j
        defocus_term = self.pf.calculate_defocus_phase(zpos)
        refocus_term = self.pf.calculate_defocus_phase(-zpos)
        imse_list = []
        sum_intensity = psf.sum()
        pupil = self.pf.mask + 0j

        for k in range(num_iters):
            # apply defocus to pupil
            pupil = pupil[None, :, :] * defocus_term
            # compute amplitude PSF
            psfa = torch.fft.ifft2(pupil)

            # compute error
            error = torch.abs(torch.abs(psfa) - obsmag)
            imse = torch.sum(error * error) / sum_intensity
            imse_list.append(imse)

            logger.info(f"iteration = {k + 1:d}, imse = {imse:12.4E}")

            # swap with observed magnitude
            psfo = obsmag * (psfa / torch.abs(psfa))

            # transform back to pupil function
            pupil2 = torch.fft.fft2(psfo)

            # refocus to z=0
            pupil2 *= refocus_term

            # average pupil
            pupil = pupil2.mean(dim=0)

            # clip regions out of suppport
            pupil *= self.pf.mask

        # clean up pupil
        pupil = torch.abs(pupil) * torch.exp(
            1j * (torch.angle(pupil) * self.pf.mask)
        )

        self.pf.pupil = pupil

        return self.pf

    def fit_otf_scaling_factor(self):
        """computes empirical gaussian 'alpha' for OTF scaling"""

        # calculate phase-retrieved otf
        phret_psf = self.pf.calculate_3d_psf(
            self.Nz, self.dz, ns=self.pf.ns, use_pupil=True
        )

        # compute cylindrical average (we don't use the z axis)
        r_freq, z_freq, obs_avg = cylindrical_average_otf(
            self.data.cpu().numpy()
        )
        r_freq, z_freq, phret_avg = cylindrical_average_otf(
            phret_psf.cpu().numpy()
        )

        # average along z-axis
        obs_1d = obs_avg.mean(axis=0)
        phret_1d = phret_avg.mean(axis=0)

        def _gaussian(k, A, a):
            arg = -(np.pi**2) * k**2
            return A * np.exp(arg / a)

        obs_ret_ratio = obs_1d / phret_1d

        # scale x axis by pixel size to get frequency in units of 1/μm
        x_data = r_freq / self.pf.dxy

        self._debug["obs_avg"] = obs_avg
        self._debug["phret_avg"] = phret_avg

        self._radial_otf_ratio = obs_ret_ratio
        self._r_freq = x_data

        popt, _ = curve_fit(_gaussian, x_data, obs_ret_ratio)

        alpha = popt[-1]

        return alpha
