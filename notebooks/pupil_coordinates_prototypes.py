# %%
import DeconTools.core.Pupil as P
import DeconTools.viz as viz
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
import torch.fft as fft

plt.rcParams["image.cmap"] = "gray"
plt.rcParams["figure.figsize"] = (4, 4)

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "qt")

# %%
device = torch.device("mps")
Ncalc = 512

c = P.OpticalConfig(Nx=Ncalc, Ny=Ncalc, dxy=0.085, NA=1.33, ni=1.525, ns=1.33)
pc = P.create_pupil_coordinates(c, 0.530)

Nz = 128
dz = 0.120
z = P.calculate_frequency_indices(Nz) * dz

yind = P.calculate_frequency_indices(256)
xind = P.calculate_frequency_indices(256)
yi, xi = torch.meshgrid(yind, xind, indexing="ij")

apodization = 1 / torch.sqrt(torch.cos(pc.theta_immersion))
# apodize mask (sine condition)
mask = (pc.mask * apodization).to(device)

# defocus aberration
z_factor = P.calculate_defocus_phase(z, pc, c, device=device)

# spherical aberration
psi = P.calculate_spherical_aberration_factor(1.0, pc, c).to(device)

# apply spherical aberration
pu = torch.where(mask > 0, mask * psi / pc.wavelength, 0.0)

# %%
# use broadcasting to apply defocus aberration to pupil for each 'slice'
pu = pu[None, :, :] * z_factor

# take the IFFT to get the amplitude PSF
psfa = fft.ifft2(pu)

# square magnitude for intensity PSF
psfi = torch.abs(psfa) ** 2

# crop from oversampled grid
psfi = psfi[:, yi, xi].cpu()
psfi /= psfi.max()
# %%
v = viz.SimpleOrthoViewer(
    fft.fftshift(psfi), norm=mcolors.PowerNorm(0.25, vmin=0, vmax=1)
)

# %%

plt.imshow(fft.fftshift(torch.abs(pu.cpu())))
