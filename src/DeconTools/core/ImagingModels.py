import gc
import math
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch
import torch.fft as fft

from .Pupil import PupilFunction


def get_device() -> torch.device:
    """
    Determine the best available device for PyTorch operations.
    Returns: torch.device for either CUDA, MPS, or CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        # fall back, "cpu" always available
        return torch.device("cpu")


def calculate_duchon_filter(real_tensor_shape: tuple[int, ...], cutoff: float):
    ndim = len(real_tensor_shape)
    eps = torch.finfo(torch.float32).eps
    freq_coords = tuple(
        torch.fft.fftfreq(s) if i < ndim else torch.fft.rfftfreq(s)
        for i, s in enumerate(real_tensor_shape, start=1)
    )

    K = torch.meshgrid(*freq_coords, indexing="ij")
    R = torch.sqrt(sum([k * k for k in K]))
    arg = torch.pi * R / cutoff + eps
    sigma = torch.sin(arg) / arg
    return sigma


class CarringtonModel:
    def __init__(
        self,
        data_shape,
        zoom_factor: float | tuple[float, ...],
        optical_params: dict[str, Any],
        device_str: str = "cpu",
        duchon_cutoff: float = 0.333,
    ):
        self.data_shape = data_shape
        self.ndim = len(data_shape)
        self.optical_params = optical_params
        self.device = torch.device(device_str)
        self.duchon_cutoff = duchon_cutoff

        if isinstance(zoom_factor, float):
            zoom_factor = (zoom_factor,) * len(data_shape)

        # calculate data-space padding

        # compute padding for linear convolution
        self.object_shape = tuple(
            int(z * s) for z, s in zip(zoom_factor, data_shape)
        )

        self.actual_zoom = tuple(
            float(o / d) for o, d in zip(self.object_shape, self.data_shape)
        )

        self.padded_object_shape = tuple(n + n - 1 for n in self.object_shape)
        self.padded_data_shape = tuple(m + m - 1 for m in self.data_shape)

        self.psf_padder = CornerPad(
            self.object_shape, self.padded_object_shape, rfft_mode=False
        )
        self.upsample_padder = CornerPad(
            self.padded_data_shape, self.padded_object_shape, rfft_mode=True
        )
        self.camera_padder = CenterPad(self.data_shape, self.padded_data_shape)
        self.output_padder = CenterPad(
            self.object_shape, self.padded_object_shape
        )
        self.calculate_object_psf()

    def calculate_object_psf(self):
        if self.ndim == 3:
            Nz, Ny, Nx = self.object_shape
            dz = self.optical_params["dz"]
            xyzoom = self.actual_zoom[1]
            zzoom = self.actual_zoom[0]

        elif self.ndim == 2:
            xyzoom = self.actual_zoom[0]
            Ny, Nx = self.object_shape

        self.pu = PupilFunction(
            Nx=Nx,
            Ny=Ny,
            NA=self.optical_params["NA"],
            ni=self.optical_params["ni"],
            ns=self.optical_params["ns"],
            dxy=self.optical_params["dxy"] / xyzoom,
            wavelength=self.optical_params["wavelength"],
        )

        psf = self.pu.calculate_3d_psf(
            Nz,
            dz / zzoom,
            ns=self.optical_params["ns"],
            device=self.device,
            scale_otf=False,
            use_pupil=False,
            source_depth=self.optical_params["source_depth"],
        )

        # pad psf
        padded_psf = self.psf_padder.forward(psf)

        self.otf = torch.fft.rfftn(padded_psf)
        sigma = calculate_duchon_filter(
            self.padded_object_shape, self.duchon_cutoff
        )
        self.icf = (sigma**2).to(self.device)

        del psf, sigma
        gc.collect()

        self.adjoint_iscale = math.prod(self.padded_object_shape) / math.prod(
            self.padded_data_shape
        )

    def forward(self, x: torch.Tensor):
        # input is in padded object space
        ft_x = torch.fft.rfftn(x)
        torch.mul(ft_x, self.otf * self.icf, out=ft_x)
        # crop to padded data space
        ft_x = self.upsample_padder.adjoint(ft_x)
        x = torch.fft.irfftn(ft_x, s=self.padded_data_shape)
        return self.camera_padder.adjoint(x)

    def adjoint(self, x: torch.Tensor):
        # input is in data-space, need to pad for linear convolution
        x = self.camera_padder.forward(x)
        # take the FT
        ft_x = torch.fft.rfftn(x)
        # move to padded object space
        ft_x = self.upsample_padder.forward(ft_x)
        torch.mul(ft_x, self.icf.conj() * self.otf.conj(), out=ft_x)
        # scale fourier coefficient to preserve total energy
        return torch.fft.irfftn(
            ft_x * self.adjoint_iscale, s=self.padded_object_shape
        )


def calculate_corner_indices(n: int, half=False) -> torch.Tensor:
    positive_indices = torch.arange(0, n // 2 + 1)
    if half:
        return positive_indices
    else:
        negative_indices = torch.arange(-(n - (n // 2 + 1)), 0)
        return torch.cat((positive_indices, negative_indices))


class CenterPad:
    """Zero-pad the input by centering it in an array of shape out_shape."""

    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, x):
        out = torch.zeros(self.out_shape, dtype=x.dtype, device=x.device)
        if len(self.in_shape) == 2:
            ny, nx = self.in_shape
            Ny, Nx = self.out_shape
            py, px = (Ny - ny) // 2, (Nx - nx) // 2
            out[py : py + ny, px : px + nx] = x
        else:
            nz, ny, nx = self.in_shape
            Nz, Ny, Nx = self.out_shape
            pz, py, px = (Nz - nz) // 2, (Ny - ny) // 2, (Nx - nx) // 2
            out[pz : pz + nz, py : py + ny, px : px + nx] = x
        return out

    def adjoint(self, y):
        if len(self.in_shape) == 2:
            ny, nx = self.in_shape
            Ny, Nx = y.shape
            py, px = (Ny - ny) // 2, (Nx - nx) // 2
            return y[py : py + ny, px : px + nx]
        else:
            nz, ny, nx = self.in_shape
            Nz, Ny, Nx = y.shape
            pz, py, px = (Nz - nz) // 2, (Ny - ny) // 2, (Nx - nx) // 2
            return y[pz : pz + nz, py : py + ny, px : px + nx]


class CornerPad:
    """corner pad for RFFT or PSF"""

    def __init__(self, in_shape, out_shape, rfft_mode: bool = True):
        self.ndim = len(in_shape)
        self.real_in_shape = in_shape
        self.real_out_shape = out_shape

        if rfft_mode:
            # in RFFT only keep half samples for the last dim.
            self.in_shape = tuple(
                s if i < self.ndim else s // 2 + 1
                for i, s in enumerate(self.real_in_shape, start=1)
            )
            self.out_shape = tuple(
                s if i < self.ndim else s // 2 + 1
                for i, s in enumerate(self.real_out_shape, start=1)
            )
        else:
            self.in_shape = in_shape
            self.out_shape = out_shape

        if self.ndim == 2:
            self.ind_y = calculate_corner_indices(self.real_in_shape[0])
            self.ind_x = calculate_corner_indices(
                self.real_in_shape[1], half=rfft_mode
            )
        elif self.ndim == 3:
            self.ind_z = calculate_corner_indices(self.real_in_shape[0])
            self.ind_y = calculate_corner_indices(self.real_in_shape[1])
            self.ind_x = calculate_corner_indices(
                self.real_in_shape[2], half=rfft_mode
            )

    def forward(self, x):
        out = torch.zeros(self.out_shape, dtype=x.dtype, device=x.device)
        if self.ndim == 2:
            ind_y = self.ind_y.to(x.device)
            ind_x = self.ind_x.to(x.device)
            out[ind_y[:, None], ind_x[None, :]] = x
        elif self.ndim == 3:
            ind_z = self.ind_z.to(x.device)
            ind_y = self.ind_y.to(x.device)
            ind_x = self.ind_x.to(x.device)
            out[
                ind_z[:, None, None],
                ind_y[None, :, None],
                ind_x[None, None, :],
            ] = x
        return out

    def adjoint(self, y):
        if self.ndim == 2:
            ind_y = self.ind_y.to(y.device)
            ind_x = self.ind_x.to(y.device)
            return y[ind_y[:, None], ind_x[None, :]]
        elif self.ndim == 3:
            ind_z = self.ind_z.to(y.device)
            ind_y = self.ind_y.to(y.device)
            ind_x = self.ind_x.to(y.device)
            return y[
                ind_z[:, None, None],
                ind_y[None, :, None],
                ind_x[None, None, :],
            ]
