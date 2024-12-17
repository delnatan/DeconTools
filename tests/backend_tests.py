# %%
import colorcet as cc
import DeconTools.core.PSF as p
import DeconTools.viz as viz
import matplotlib.colors as mcolors
import numpy as np
import scipy.fft as fft
import tifffile
from DeconTools.operators.fftops import LinearOperator

# %%
data = tifffile.imread(
    "/Users/delnatan/Desktop/scratch/moisan_ft_ptype/cells3d.tif"
)
Nz, Nch, Ny, Nx = data.shape
ch01 = data[:, 0].astype(np.float32)
ch02 = data[:, 1]

# %%
params = p.MicroscopeParameters(
    0.530,
    numerical_aperture=1.40,
    sample_refractive_index=1.33,
    immersion_refractive_index=1.515,
    pixel_size=0.065,
)
pu = p.PupilFunction(params, Nx=Nx, Ny=Ny)
zplanes = p.compute_z_planes(Nz, dz=0.290)
psf3d = pu.calculate_3d_psf(zplanes, zpos=-1.0)

# %%
v = viz.SimpleOrthoViewer(
    psf3d, cmap=cc.m_gouldian, norm=mcolors.PowerNorm(0.5)
)
v.show()
# %%
import torch

mps = torch.device("mps")
torch_psf3d = torch.from_numpy(psf3d.astype(np.float32)).to(mps)
torch_img = torch.from_numpy(ch01.astype(np.float32)).to(mps)

H = LinearOperator(torch_psf3d, shape=(Nz, Ny, Nx))

res = H.adjoint(torch_img)

# %% test finite difference kernel
dxx = np.zeros((3, 3, 3), dtype=np.float32)
dxx[0, 0, 0] = -2
dxx[0, 0, 1] = 1
dxx[0, 0, -1] = 1

torch_dxx = torch.from_numpy(dxx).to(mps)
Dxx = LinearOperator(torch_dxx, shape=(Nz, Ny, Nx))
resxx = Dxx.dot(torch_img)

dzz = np.zeros((3, 3, 3), dtype=np.float32)
dzz[0, 0, 0] = -2
dzz[1, 0, 0] = 1
dzz[-1, 0, 0] = 1

torch_dzz = torch.from_numpy(dzz).to(mps)
Dzz = LinearOperator(torch_dzz, shape=(Nz, Ny, Nx))
reszz = Dzz.dot(torch_img)
