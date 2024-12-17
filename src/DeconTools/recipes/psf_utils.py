from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter

from ..core.PSF import MicroscopeParameters, PupilFunction, compute_z_planes
from ..operators.fftops import LinearOperator


def distill_psf_from_data(
    data: np.ndarray,
    microscope_parameters: MicroscopeParameters,
    dz: float,
    threshold: float = 1000,
    tol: float = 1e-2,
    smooth: float = 1.2,
    window_max: Tuple[int] = (5, 5, 5),
    baseline_clamp: float = 100.0,
    gpu: bool = False,
    max_iter: int = 50,
) -> np.ndarray:
    """'distill' PSF from well-separated bead images like Huygen PSF tool

    The procedure is simple and it's just the 'reversed' Richardson-Lucy
    iteration. In the RL iteration, we know the PSF and we solve the 'object'
    given an observed data. We simply 'swap' the PSF and the object.

    Overloading the notation H as the linear operator and 'f' is what we want
    to solve. Given data, d, the iteration is:

    fᵏ⁺¹ = fᵏ .* Hᵗ(d ./ Hfᵏ)

    In the /forward/ Richardson-Lucy, H is the PSF or blurring operator.
    In the /reversed/ Richardson-Lucy, H is the 'object', a collection of
    delta functions (single-pixel objects).

    To generate such objects, we can do a simple thresholding (preceded by a
    simple Gaussian filtering step to reduce noise), and find the center of
    mass of each diffraction-limited spots, then create a 'point-source' image
    given the center positions of these objects.

    Arguments:
    data (np.ndarray): 3D image of well-separated beads
    microscope_parameters (MicroscopeParameters): microscope parameters
    dz (float): spacing in z-dimension. Spacing in 2D is given in
    microscope_parameters
    tol (float, optional): convergence tolerance
    smooth (float): gaussian smoothing applied to data for maxima identification
    baseline_clamp (float): baseline value for camera where data was captured
    """
    if smooth > 1e-3:
        wrk_data = gaussian_filter(data, smooth)
    else:
        wrk_data = data

    footprint = np.ones(window_max, dtype=bool)

    max_result = maximum_filter(wrk_data, footprint=footprint)

    maxlocs = np.argwhere((max_result == wrk_data) & (wrk_data > threshold))

    npeaks = len(maxlocs)
    print(f"Found {npeaks:d} peaks.")

    # create objects (delta function images)
    Nz, Ny, Nx = wrk_data.shape
    object = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    object[maxlocs[:, 0], maxlocs[:, 1], maxlocs[:, 2]] = 1.0

    # compute initial PSF
    pu = PupilFunction(microscope_parameters, Nx=Nx, Ny=Ny)
    zplanes = compute_z_planes(Nz, dz)
    psf = pu.calculate_3d_psf(zplanes)
    data0 = np.maximum(data - baseline_clamp, 0.0)

    if gpu:
        gpu = torch.device("mps")
        _object = torch.from_numpy(object.astype(np.float32)).to(gpu)
        _psf = torch.from_numpy(psf.astype(np.float32)).to(gpu)
        _data0 = torch.from_numpy(data0.astype(np.float32)).to(gpu)
        H = LinearOperator(_object, (Nz, Ny, Nx))
    else:
        H = LinearOperator(object, (Nz, Ny, Nx))
        _object = object
        _psf = psf
        _data0 = data0

    converged = False
    iteration = 0

    while not converged and iteration < max_iter:
        iteration += 1
        model = H.dot(_psf)
        ratio = _data0 / model
        update = H.adjoint(ratio)
        prev_psf = _psf * 1.0

        _psf *= update

        if gpu:
            diff_norm = torch.linalg.vector_norm(_psf - prev_psf)
            print(f"debug: {diff_norm.item()}")
            prev_norm = torch.linalg.vector_norm(prev_psf)
            print(f"debug: {prev_norm.item()}")
            rel_change = diff_norm.item() / prev_norm.item()
        else:
            rel_change = np.linalg.norm(_psf - prev_psf) / np.linalg.norm(
                prev_psf
            )

        print(
            f"\riteration = {iteration:d}, relative change = {rel_change}",
            end="",
        )

        if rel_change < tol:
            converged = True

    print("\n✓ iteration done.")

    if gpu:
        return _psf.detach().cpu().numpy()
    else:
        return _psf
