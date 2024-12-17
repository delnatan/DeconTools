# %%
import colorcet as cc
import DeconTools.core.PSF as P
import DeconTools.operators.derivatives as D
import DeconTools.viz as V
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy.ndimage as ndi

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "qt")
# %%

Nz, Ny, Nx = 64, 256, 256
hz, hy, hx = Nz / 2.0, Ny / 2.0, Nx / 2.0
ogrid = np.mgrid[-hz:hz, -hy:hy, -hx:hx]
R = np.sqrt(sum(np.power(o, 2) for o in ogrid))
shell = (R >= 22) * (R <= 24)
shell = shell.astype(float)

# %% compute PSF
mpars = P.MicroscopeParameters(
    excitation_wavelength=0.488,
    emission_wavelength=0.529,
    numerical_aperture=0.95,
    sample_refractive_index=1.36,
    immersion_refractive_index=1.0,
    pixel_size=0.1043,
)
dz = 0.3
zplanes = P.compute_z_planes(Nz, dz)
pu = P.PupilFunction(mpars, Nx=Nx, Ny=Ny, oversampling=4)
psf3d = pu.calculate_3d_psf(zplanes, -1.0)

V.SimpleOrthoViewer(
    fft.fftshift(psf3d),
    norm=mcolors.PowerNorm(0.15, vmin=0, vmax=1),
    cmap=cc.m_gouldian,
)
# %%
otf3d = fft.rfftn(psf3d)
ft_shell = fft.rfftn(shell)
_conv = ft_shell * otf3d
blurred = fft.irfftn(ft_shell * otf3d, axes=(0, 1, 2), s=(Nz, Ny, Nx))

# crop in Fourier domain to simulate undersampling
uNy, uNx = 128, 128
hy, hx = uNy // 2, uNx // 2
hy = np.r_[:hy, -hy:0]
hx = np.r_[:hx, -hx:0]
yi, xi = np.ix_(hy, hx)

blurred_undersampled = fft.irfftn(
    _conv[:, yi, xi], axes=(0, 1, 2), s=(Nz, uNy, uNx)
)

# %%
ftaxes = tuple(i for i in range(3))
filters = D.compute_3D_second_order_derivative_filters((Nz, Ny, Nx))
sfilters = np.stack(filters)
self_adjoint_filter = np.conj(sfilters) * sfilters


D_shell = fft.irfftn(
    ft_shell[None, ...] * sfilters, axes=(1, 2, 3), s=(Nz, Ny, Nx)
)
sj_D_shell = sum(
    fft.irfftn(
        ft_shell[None, ...] / (self_adjoint_filter + 1e-6),
        axes=(1, 2, 3),
        s=(Nz, Ny, Nx),
    )
)


# %%
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))

ax[0, 0].imshow(Dshell[0][:, 64, :])  # zz
ax[0, 1].imshow(Dshell[1][30, :, :])  # yy
ax[0, 2].imshow(Dshell[2][30, :, :])  # xx
ax[1, 0].imshow(Dshell[3][:, :, 64])  # zy
ax[1, 1].imshow(Dshell[4][:, 64, :])  # zx
ax[1, 2].imshow(Dshell[5][30, :, :])  # yx
