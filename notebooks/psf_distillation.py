# %%
from pathlib import Path
from turtle import clearstamps

import matplotlib.colors as mcolors
import numpy as np
import tifffile
from DeconTools.recipes.psf_utils import distill_3D_psf
from DeconTools.viz import SimpleOrthoViewer
from imaris_file import ImarisReader

# %%
root = Path("/Users/delnatan/StarrLuxtonLab/Experiments/imaging/yr2024/PSFs")

psfs = []

flist = [
    "ps_air_cc170_wf.ims",
    "ps_air_cc170_wf_1.ims",
    "ps_air_cc170_wf_2.ims",
]

for fn in flist:
    with ImarisReader(root / fn) as ims:
        ims.info()
        data = ims.get_data()
        Nz = data.shape[0]
        psf = distill_3D_psf(data, check=False)
        psfs.append(psf)


# %% average all 3 distilled PSFs
def clamp_and_normalize(x):
    wrk = np.maximum(x - 100.0, 0.0)
    return wrk / wrk.sum()


avg_psf = sum(clamp_and_normalize(p) / 3.0 for p in psfs)

# %%
import tifffile

metadata = {
    "axes": "ZYX",
    "PhysicalSizeX": 0.1565,
    "PhysicalSizeY": 0.1565,
    "PhysicalSizeZ": 0.381,
    "ChannelWavelengths": [529],
}

tifffile.imwrite(
    "/Users/delnatan/StarrLuxtonLab/Experiments/PSFs/40x_air_wv_529nm.ome.tif",
    avg_psf,
    metadata=metadata,
)
