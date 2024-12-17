# %%
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
import scipy.ndimage as ndi
import tifffile
from DeconTools.core import PSF as p
from matplotlib.colors import PowerNorm

# %%


# psf info (0.1565, 0.1565, 0.381)
# λ_em, 529 nm
def pad_to_size(array, target_shape):
    diff = np.subtract(target_shape, array.shape)
    pad_width = [(d // 2, d - d // 2) for d in diff]
    return np.pad(array, pad_width, mode="constant", constant_values=0)


wf_psf = tifffile.imread(
    "/Users/delnatan/StarrLuxtonLab/Experiments/imaging/"
    "yr2024/scripts/averaged_40x_air_PSF_wf.tif"
)

wf_psf /= wf_psf.max()

padded_psf = pad_to_size(wf_psf, (19, 128, 128))
Nz, Ny, Nx = padded_psf.shape


# %% determine PSF center
def m1(y):
    x = np.arange(y.size)
    p = y / y.sum()
    return np.sum(p * x)


cx = m1(padded_psf.max(axis=(0, 1)))
cy = m1(padded_psf.max(axis=(0, 2)))
sx = (128 / 2.0) - cx
sy = (128 / 2.0) - cy

centered_psf = ndi.shift(padded_psf, (0, sy, sx))

# %% define z-planes
zprof = padded_psf.max(axis=(1, 2))
cz = m1(zprof)
zplanes = np.arange(Nz) - cz
zplanes *= 0.381  # convert to microns

central_slice = centered_psf[:, 64, :]

fig, ax = plt.subplots()

ax.imshow(central_slice, norm=PowerNorm(0.15, vmin=0), origin="lower")

# %%
params = p.MicroscopeParameters(
    excitation_wavelength=0.448,
    emission_wavelength=0.529,
    immersion_refractive_index=1.0,
    sample_refractive_index=1.0,
    pixel_size=0.1565,
    numerical_aperture=0.95,
)
pu = p.PupilFunction(params, Nx=Nx, Ny=Ny)
pupil0 = pu.calculate_ideal_pupil()

fig, ax = plt.subplots(figsize=(8, 4), ncols=2)
ax[0].imshow(pupil.real, vmin=0, vmax=1)
ax[1].imshow(pupil.imag)

# %% phase retrieval via Gerchberg-Saxton algorithm
amplitudes = fft.fftshift(
    np.sqrt(np.maximum(centered_psf[5:-5, :, :], 0.0)), axes=(1, 2)
)
wrk_zplanes = zplanes[5:-5]

k0sq = (
    pu.params.immersion_refractive_index / pu.params.emission_wavelength
) ** 2
kz = np.sqrt(k0sq - pu.R**2 + 0j)


outside_support = pu.R > (
    pu.params.numerical_aperture / pu.params.emission_wavelength
)
within_support = ~outside_support

defocus_term = 1j * 2.0 * np.pi * wrk_zplanes[:, None, None] * kz[None, :, :]
_defocus = np.exp(defocus_term)
_focus = np.exp(-defocus_term)

it = 0
iMSElist = []

pupil = pupil0 * within_support
sum_intensity = np.sum(amplitudes**2)

while it < 500:
    it += 1
    g = pupil[None, :, :] * _defocus
    psfa = fft.ifft2(g, axes=(1, 2))
    error = np.abs(np.abs(psfa) - amplitudes)
    error2 = np.sum(error * error)
    iMSE = error2 / sum_intensity
    print(f"iteration = {it:d}, iMSE = {iMSE:14.4E}")
    iMSElist.append(iMSE)

    # swap magnitudes by unit projection
    _unit = psfa / (np.abs(psfa) + 1e-14)
    wrk = amplitudes * _unit
    gprime = fft.fft2(wrk)
    # refocus
    gprime = gprime * _focus
    gprime_mean = gprime.mean(axis=0)

    pupil = gprime_mean * within_support

plt.plot(iMSElist, "k.")
# %%
fig, ax = plt.subplots(figsize=(8, 4), ncols=2)
ax[0].imshow(fft.fftshift(np.abs(fpupil)), cmap=cc.m_gouldian)
ax[1].imshow(
    fft.fftshift(np.angle(fpupil) * within_support),
    cmap=cc.m_gouldian,
    vmin=-np.pi / 2,
    vmax=np.pi / 2,
)
ax[0].set_title("Pupil magnitude")
ax[1].set_title("Pupil phase")
# %% compare with data
from matplotlib.ticker import FuncFormatter, MultipleLocator

dz = 0.381
pixel_per_um = 1 / dz

# bead correction
fy = fft.fftfreq(Ny, d=pu.params.pixel_size)
fx = fft.fftfreq(Nx, d=pu.params.pixel_size)
fz = fft.fftfreq(Nz, d=0.381)
_kz, _ky, _kx = np.meshgrid(fz, fy, fx, indexing="ij")
R = np.sqrt(_kx**2 + _ky**2 + _kz**2)
arg = np.pi * R * 0.175
bead_ff = 3 * (np.sin(arg) / (arg**3 + 1e-14) - np.cos(arg) / (arg**2 + 1e-14))


def pixel_to_um(x, pos):
    return f"{x / pixel_per_um:.1f}"


formatter = FuncFormatter(pixel_to_um)


fpupil = pupil / np.sqrt(np.cos(pu.theta_1))

obs_defocus = 1j * np.pi * 2.0 * zplanes[:, None, None] * kz[None, :, :]
obs_defocus_term = np.exp(obs_defocus)
ret_psfa = fft.ifft2(fpupil * obs_defocus_term, axes=(1, 2))

ret_psfi = np.abs(ret_psfa) ** 2
ret_otf = fft.fftn(ret_psfi) * bead_ff
ret_psfi = fft.ifftn(ret_otf).real
ret_psfi = fft.ifftshift(ret_psfi, axes=(1, 2))

ret_central_slice = ret_psfi[:, 64, :]

fig, ax = plt.subplots(nrows=2, figsize=(6, 7), constrained_layout=True)
ax[0].imshow(central_slice, norm=PowerNorm(0.2), aspect=2.34, origin="lower")
ax[1].imshow(
    ret_central_slice, norm=PowerNorm(0.2), aspect=2.34, origin="lower"
)
ax[0].yaxis.set_major_formatter(formatter)
ax[0].yaxis.set_major_locator(MultipleLocator(pixel_per_um * 2.5))
ax[0].set_ylabel("Relative z-position, $\mu m$", fontsize=9)
