# %%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from DeconTools.core.Pupil import (
    PhaseRetrievedPupilFunction,
    PupilFunction,
    calculate_frequency_indices,
    calculate_sphere_form_factor,
)
from DeconTools.viz import SimpleOrthoViewer

# %%
with tifffile.TiffFile(
    "/Users/delnatan/StarrLuxtonLab/Experiments/PSFs/40x_air_wv_529nm.ome.tif"
) as tif:
    metadata = tif.ome_metadata
    image = tif.asarray()
    timage = torch.from_numpy(image)
    Nz, Ny, Nx = image.shape

# %%
prob = PhaseRetrievedPupilFunction(
    timage, 0.1565, 0.381, 0.529, 0.95, 1.0, 1.0
)

pu = prob.retrieve_phase()
# %%
plt.imshow(torch.fft.fftshift(torch.angle(pu.pupil)), cmap="plasma")
plt.show()

# %%
retpsf = pu.calculate_3d_psf(Nz, 0.381, ns=1.0, use_pupil=True)

retotf = torch.fft.rfftn(retpsf)
expotf = torch.fft.rfftn(timage)

retotfi = torch.abs(retotf) ** 2
expotfi = torch.abs(expotf) ** 2

# %%

# %%
otfratio = retotfi / expotfi
v = SimpleOrthoViewer(
    torch.fft.fftshift(otfratio, dim=(0, 1)),
    norm=mcolors.PowerNorm(0.5, vmin=0, vmax=50),
)
# %%
fig, ax = plt.subplots(nrows=2, figsize=(8, 6))
ax[0].imshow()
# %%
v = SimpleOrthoViewer(
    torch.fft.fftshift(expotfi, dim=(0, 1)), norm=mcolors.LogNorm()
)
v.show()
# %%
psf = prob.apply_bead_size_correction(0.2)
v = SimpleOrthoViewer(torch.fft.fftshift(psf), norm=mcolors.PowerNorm(0.25))
v.show()
# %%
v = SimpleOrthoViewer(
    torch.fft.fftshift(spsf), norm=mcolors.PowerNorm(0.15), cmap="plasma"
)
# %%
pu = PupilFunction(
    Nx=Nx, Ny=Ny, NA=0.95, ni=1.0, ns=1.0, dxy=0.1565, wavelength=0.529
)

# %%

# calculate focal plane positions
dz = 0.381
z = calculate_frequency_indices(Nz) * dz
zidx = torch.argwhere(torch.abs(z) < 3).squeeze()
max_iter = 40
# experimental magnitude from sqrt(intensity)
psf = spsf[zidx]
obsmag = torch.sqrt(psf)
zpos = z[zidx]
pupil = pu.mask + 0j
# phase term for applying defocus to pupil
defocus_term = pu.calculate_defocus_phase(zpos)
# refocusing term for pupil to undo defocus term
refocus_term = pu.calculate_defocus_phase(-zpos)
sum_intensity = psf.sum()
imse_list = []

for k in range(max_iter):
    # calculate defocus term
    defocus_term = pu.calculate_defocus_phase(zpos)
    # apply to pupil
    pupil = pupil[None, :, :] * defocus_term
    # compute complex point-spread function (amplitude PSF)
    psfa = torch.fft.ifft2(pupil)

    # compute error
    error = torch.abs(torch.abs(psfa) - obsmag)
    imse = torch.sum(error * error) / sum_intensity
    print(f"iter. = {k + 1}, imse = {imse:14.6E}")
    imse_list.append(imse)

    psfo = obsmag * (psfa / torch.abs(psfa))
    # transform back into pupil
    pupil2 = torch.fft.fft2(psfo)
    # refocus pupil
    pupil2 *= refocus_term
    # average pupil
    pupil = pupil2.mean(dim=0)
    # clip regions out of support
    pupil *= pu.mask

# clean up pupil
pupil = torch.abs(pupil) * torch.exp(1j * (torch.angle(pupil) * pu.mask))
# %%
fig, ax = plt.subplots(figsize=(8, 4), ncols=2)
ax[0].imshow(torch.fft.fftshift(torch.abs(pupil)), cmap="plasma")
ax[1].imshow(torch.fft.fftshift(torch.angle(pupil)), cmap="plasma")
plt.show()

# %% reassign pupil
pu.pupil = pupil

# %%

retpsf = pu.calculate_3d_psf(
    Nz=60, dz=0.2, device="mps", use_pupil=True, source_depth=2.0
)
retpsf = retpsf.cpu()

# %%
v = SimpleOrthoViewer(
    torch.fft.fftshift(retpsf), norm=mcolors.PowerNorm(0.25), cmap="plasma"
)
v.show()
