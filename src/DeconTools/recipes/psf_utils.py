"""
various tools for extracting PSFs from raw z-stacks

"""

from typing import Tuple

import numpy as np
import torch
import torch.fft as fft
from scipy.ndimage import gaussian_laplace, maximum_filter

from ..utils.padding import OriginPad, SpatialPad


def summarize_vector(vec, percentiles=[95, 98, 99, 99.9, 99.99, 100]):
    stats = np.percentile(vec, percentiles)
    summary = "\n".join(
        f"{p:>5}%: {s:.3f}" for p, s in zip(percentiles, stats)
    )
    return f"Summary of intensity values:\n{summary}"


def distill_3D_psf(
    image: np.ndarray,
    baseline_clamp: float = 100.0,
    num_iter: int = 25,
    gaussian_sigma: float = 1.4,
    maxima_domain_size: int = 11,
    psf_yx_shape: tuple[int, int] = (256, 256),
    int_threshold: float = 1000.0,
    check: bool = False,
    device_str: str = "mps",
):
    if image.ndim == 3:
        Nz = int(image.shape[0])
        psf_shape = (Nz,) + psf_yx_shape
    else:
        Nz = None
        psf_shape = psf_yx_shape

    data = np.maximum(image.astype(np.float32) - baseline_clamp, 0.0)

    # detect PSF locations
    LoG_img = -gaussian_laplace(data, gaussian_sigma)
    _s = maxima_domain_size
    psf_domain = np.ones((_s, _s, _s))
    maxfilt_img = maximum_filter(LoG_img, footprint=psf_domain)

    if check:
        intensities = data[maxfilt_img == LoG_img]
        Npeaks = len(intensities)

        summary_str = summarize_vector(intensities)
        print(summary_str)

        peaklocs = np.argwhere(
            (LoG_img == maxfilt_img) & (data > int_threshold)
        )

        print(f"Found {len(peaklocs)} peaks with threshold {int_threshold}.")

        for i in data[tuple(peaklocs.T)]:
            print(f"{i:.1f}")

        return None

    peaklocs = np.argwhere((LoG_img == maxfilt_img) & (data > int_threshold))

    peakintensities = data[tuple(peaklocs.T)]
    init_intensity = float(peakintensities.mean())

    # instantiate PyTorch device for computation
    device = torch.device(device_str)

    point_sources = torch.zeros(
        image.shape, dtype=torch.float32, device=device
    )

    point_sources[tuple(peaklocs.T)] = torch.from_numpy(peakintensities).to(
        device
    )

    # determine padding sizes for linear convolution
    target_shape = tuple(m + n - 1 for m, n in zip(image.shape, psf_shape))

    border_padder = SpatialPad(image.shape, target_shape)
    origin_padder = OriginPad(psf_shape, target_shape)

    # begin iterative reverse Richardson-Lucy process
    # pad arrays to emulate linear convolution
    known_object = border_padder.pad(point_sources)
    padded_data = border_padder.pad(torch.from_numpy(data).to(device))

    ft_object = fft.rfftn(known_object)
    psf_estimate = torch.ones(target_shape, device=device) * init_intensity
    sigma2 = 1.0

    for k in range(num_iter):
        print(
            f"\rDistilling iteration, k = {k + 1:3d} / {num_iter}...", end=""
        )
        otf = fft.rfftn(psf_estimate)
        model = fft.irfftn(ft_object * otf, s=target_shape)
        model = torch.clamp(model, min=0.0)

        ratio = (padded_data + sigma2) / (model + sigma2)
        update = fft.irfftn(
            ft_object.conj() * fft.rfftn(ratio), s=target_shape
        )
        psf_estimate = torch.where(
            update >= 0, psf_estimate * update, psf_estimate
        )

    print("\n>>Done.")

    out = origin_padder.unpad(psf_estimate, frequency=False)

    return out.cpu().numpy()
