# %%
import DeconTools.viz as viz
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from DeconTools.core import PSF as p

# %%
params = p.MicroscopeParameters(
    wavelength=0.485,
    numerical_aperture=0.95,
    pixel_size=0.100,
    immersion_refractive_index=1.0,
    sample_refractive_index=1.33,
)

pupil = p.PupilFunction(params, Ny=128, Nx=128)
zplanes = p.compute_z_planes(128, 0.1)

actual_pos = np.arange(10) * 0.25

psf4d = np.zeros((10, 128, 128, 128))

for i, zp in enumerate(actual_pos):
    psf3d = pupil.calculate_3d_psf(zplanes, zp)
    ipsf3d = fft.fftshift(psf3d)
    psf4d[i] = ipsf3d

# %%
import tifffile

tifffile.imwrite("psf4d_air.tif", psf4d.astype(np.float32))

# %% look at position shift
plt.plot(actual_pos, measured_pos, "m-")
plt.plot(actual_pos, actual_pos, "r--")
plt.show()
# %%
v = viz.SimpleOrthoViewer(
    ipsf3d, norm=mcolors.PowerNorm(0.25, vmin=0, vmax=0.8)
)
v.show()
