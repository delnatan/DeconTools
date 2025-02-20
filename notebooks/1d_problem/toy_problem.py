import numpy as np
import numpy.fft as fft
import torch
from scipy.special import j1


def lorentzian(x, x0, gamma, amplitude):
    """Lorentzian function"""
    return amplitude * gamma**2 / ((x - x0) ** 2 + gamma**2)


def gaussian_kernel_1d(n, sigma, fourier=True):
    """Gaussian kernel

    Real-space kernel:
    g(x) = exp(-0.5 * x² / σ²)

    Fourier-space kernel:
    G(X) = exp(-2 * π² * σ² * X²)

    """
    if fourier:
        k = fft.rfftfreq(n)
        ft_kernel = np.exp(-2 * (np.pi * sigma * k) ** 2)
        return ft_kernel
    else:
        x = fft.fftfreq(n) * n
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel


def airy_1d(N, dxy, NA=1.4, wavelength=0.530):
    """
    simulate 1D PSF
    """
    eps = np.finfo(np.float32).eps
    scale = NA / (wavelength / dxy)
    r = np.fft.fftfreq(N) * N * scale
    R = np.sqrt(r**2)
    airy = 2 * j1(2 * np.pi * R + eps) / (2 * np.pi * R + eps)
    return airy**2


def make_nonuniform_background(n, magnitude=5.0, correlation_length=20.0):
    # Start with random walk
    rw = np.cumsum(np.random.normal(loc=0, scale=magnitude, size=n))
    # Create Gaussian filter
    smooth_kernel = gaussian_kernel_1d(n, correlation_length, fourier=False)
    # Filter and add offset to keep positive
    background = np.convolve(rw, smooth_kernel, mode="same")
    background = background - background.min() + 100
    return background


class ToyModel:
    def __init__(
        self,
        data_size: int,
        zoom: float,
        pad: int,
        icf_sigma: float,
        otf_sigma: float,
    ):
        self.zoom = zoom
        self.pad = pad
        # data shape
        self.M = data_size + 2 * pad
        # hidden shape
        self.N = int(zoom * self.M)

        # fourier data index
        self.ft_data_index = (torch.fft.rfftfreq(self.M) * self.M).type(
            torch.int64
        )

        self.ft_hidden_size = self.N // 2 + 1

        self.ICF = torch.from_numpy(
            gaussian_kernel_1d(self.N, icf_sigma, fourier=True)
        ).type(torch.float32)

        self.OTF = torch.from_numpy(
            gaussian_kernel_1d(self.N, otf_sigma, fourier=True)
        ).type(torch.float32)

        self.unpad_data_indices = slice(self.pad, -self.pad)

        zoom_pad = int(self.pad * zoom)
        self.unpad_object_indices = slice(zoom_pad, -zoom_pad)

    def info(self):
        infostr = ""
        infostr += f"M = {self.M}\n"
        infostr += f"N = {self.N}\n"
        infostr += f"pad = {self.pad}\n"
        infostr += f"Cᴺ = {self.ft_hidden_size}\n"
        print(infostr)

    def forward(self, x):
        """forward transform: hidden-space to data-space

        'extended to compact'
        """
        ft_x = torch.fft.rfft(x)
        torch.mul(ft_x, self.ICF, out=ft_x)  # * ICF
        torch.mul(ft_x, self.OTF, out=ft_x)  # * OTF
        crop_ft_d = ft_x[self.ft_data_index]  # * Fourier crop
        crop_d = torch.fft.irfft(crop_ft_d, n=self.M)
        return crop_d[self.pad : -self.pad]  # * unpad

    def adjoint(self, y):
        # extended input boundaries
        ypad = torch.zeros(self.M)
        ypad[self.pad : -self.pad] = y  # pad

        ft_ypad = torch.fft.rfft(ypad)  # Fourier pad
        ft_xpad = torch.zeros(self.ft_hidden_size, dtype=torch.complex64)
        ft_xpad[self.ft_data_index] = ft_ypad[self.ft_data_index]

        torch.mul(ft_xpad, torch.conj(self.OTF), out=ft_xpad)  # * OTF
        torch.mul(ft_xpad, torch.conj(self.ICF), out=ft_xpad)  # * ICF

        scale = self.N / self.M
        return torch.fft.irfft(ft_xpad * scale, n=self.N)
