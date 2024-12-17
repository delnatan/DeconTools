# %%
import DeconTools.core.PSF as p
import DeconTools.viz as viz
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from DeconTools.operators.fftops import CompositeLinearOperator, LinearOperator

# %%
# psf2d = tifffile.imread("../data/mito_psf2d.tif").astype(float)
mpars = p.MicroscopeParameters(0.485, 0.95, 1.36, 1.0, 0.156)
data = tifffile.imread(
    "/Users/delnatan/BurgessLab/" "yeast_nuc_procs/dapi_img_crop.tif"
).astype(np.float32)
data = np.maximum(data - 100.0, 0.0)
Nz, Ny, Nx = data.shape
pu = p.PupilFunction(mpars, Ny=Ny, Nx=Nx)
zplanes = p.compute_z_planes(Nz, 0.581)
psf3d = pu.calculate_3d_psf(zplanes, zpos=-2.0)
psf3d = psf3d ** 2
psf3d /= psf3d.sum()

H = LinearOperator(psf3d, shape=data.shape)
# construct 2nd order finite-difference operators
kernel_zz = np.zeros((3, 3, 3))
kernel_zz[0, 0, 0] = -2
kernel_zz[-1, 0, 0] = 1
kernel_zz[1, 0, 0] = 1

kernel_yy = np.zeros((3, 3, 3))
kernel_yy[0, 0, 0] = -2
kernel_yy[0, -1, 0] = 1
kernel_yy[0, 1, 0] = 1

kernel_xx = np.zeros((3, 3, 3))
kernel_xx[0, 0, 0] = -2
kernel_xx[0, 0, -1] = 1
kernel_xx[0, 0, 1] = 1

kernel_zy = np.zeros((2, 2, 2))
kernel_zy[0, 0, 0] = 1
kernel_zy[-1, 0, 0] = -1
kernel_zy[0, -1, 0] = -1
kernel_zy[-1, -1, 0] = 1

kernel_yx = np.zeros((2, 2, 2))
kernel_yx[0, 0, 0] = 1
kernel_yx[0, -1, 0] = -1
kernel_yx[0, 0, -1] = -1
kernel_yx[0, -1, -1] = 1

kernel_zx = np.zeros((2, 2, 2))
kernel_zx[0, 0, 0] = 1
kernel_zx[-1, 0, 0] = -1
kernel_zx[0, 0, -1] = -1
kernel_zx[-1, 0, -1] = 1


lta_ratio = 0.156 / 0.585

Dxx = LinearOperator(kernel_xx, data.shape)
Dyy = LinearOperator(kernel_yy, data.shape)
Dzz = LinearOperator(lta_ratio**2 * kernel_zz, data.shape)
Dzy = LinearOperator(kernel_zy, data.shape)
Dyx = LinearOperator(kernel_yx, data.shape)
Dzx = LinearOperator(kernel_zx, data.shape)

sqrt2 = np.sqrt(2.0)
D = CompositeLinearOperator(
    [Dzz, Dyy, Dxx, Dzy, Dzx, Dyx],
    weights=[1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2],
)

# %%
v = viz.SimpleOrthoViewer(np.fft.fftshift(psf3d), norm=mcolors.PowerNorm(0.15))
v.show()
# %% gradient expressions


def KL(f, D, H):
    # ∑D * log(D/Hf) - (D - Hf)
    Hf = np.maximum(H.dot(f), 1e-7)
    return np.sum(D * np.log(D / Hf) - (D - Hf))


def S(f, m):
    # ∑f * log(f/m) - (f - m)
    return np.sum(f * np.log(f / m) - (f - m))


def gradKL(f, D, H):
    # Hᵗ(D/Hf) + 1
    Hf = np.maximum(H.dot(f), 1e-7)
    ratio = D / Hf
    g = H.adjoint(ratio) + 1.0
    return g


def gradS(f, m):
    return np.log(f / m)


# %%
prior = np.ones_like(data) * 1e-3
f = prior.copy()

# %%
%matplotlib qt
# plt.ioff()

# %%

for k in range(10):
    Hf = np.maximum(H.dot(f), 1e-7)
    ratio = data / Hf
    update = H.adjoint(ratio)
    λ = 0.001
    sgrad = gradS(f, prior)
    # rgrad = D.gradient(f)
    f = (f / (1 + λ * sgrad)) * update

# %%
clo, chi = np.percentile(f, (0.1, 99.99))
v = viz.SimpleOrthoViewer(
    f, norm=mcolors.PowerNorm(0.75, vmin=clo, vmax=chi)
)
v.show()
