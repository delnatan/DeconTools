# %%
import colorcet as cc
import DeconTools.core.PSF as p
import DeconTools.operators.fftops as f
import DeconTools.viz as v
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
import scipy.ndimage as ndi
import torch
from DeconTools.recipes.psf_utils import distill_psf_from_data
from imaris_file import ImarisReader
from matplotlib.colors import PowerNorm

get_ipython().run_line_magic("matplotlib", "qt")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%
fhd = ImarisReader("/Volumes/DE_extHD01/BC43/ps-speckYG_1.ims")
fhd.info()
data = fhd.get_data()[:, 175 : 175 + 1024, 475 : 475 + 1024]
Nz, Ny, Nx = data.shape

fhd.close()
# %%
params = p.MicroscopeParameters(
    excitation_wavelength=0.488,
    emission_wavelength=0.529,
    numerical_aperture=1.42,
    pixel_size=0.1043,
    sample_refractive_index=1.47,
    immersion_refractive_index=1.515,
)
# %%
retpsf = distill_psf_from_data(
    data, params, 0.1961, 2500, max_iter=20, gpu=False
)
# %% visualize 2d otf at central slice

retpsf /= retpsf.max()

retotf = fft.fftn(retpsf)

fig, ax = plt.subplots(constrained_layout=True)

v.visualize_central_otf(retotf, params, ax, norm=PowerNorm(0.15))

plt.show()
# %%
pu = p.PupilFunction(Nx=Nx, Ny=Ny, params=params)
zplanes = p.compute_z_planes(Nz, 0.381)
psf0 = pu.calculate_3d_psf(zplanes)

# %%


# %%
viewer = v.SimpleOrthoViewer(fft.fftshift(psf0), norm=PowerNorm(0.15))
viewer.show()

# %%
labels, nfeats = ndi.label(data > 2500)
com = ndi.center_of_mass(data > 2500, labels=labels, index=np.arange(nfeats))

object = np.zeros((Nz, Ny, Nx))
nzindices = np.array([np.array(s).astype(int) for s in com[1:]])

# %%
object[nzindices[:, 0], nzindices[:, 1], nzindices[:, 2]] = 1.0

H = f.LinearOperator(object, (Nz, Ny, Nx))
data0 = np.maximum(data - 100.0, 0.0)
psf = psf0.copy()

for k in range(50):
    print(f"\rIteration = {k}", end="")
    Hf = H.dot(psf)
    ratio = data0 / Hf
    update = H.adjoint(ratio)
    psf *= update

print("")
