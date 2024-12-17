import numpy as np
import torch
import torch.fft
from scipy.special import lambertw


def prox_poisson(ν, ρ, b):
    """
    Args:
        ν (torch.Tensor): input tensor
    """
    arg = (1 - ρ * ν) / ρ / 2.0
    arg2 = torch.maximum(arg * arg + b / ρ, torch.tensor(0.0, device=ν.device))
    return -arg + torch.sqrt(arg2)


def prox_indicator(ν):
    return torch.maximum(ν, torch.tensor(0.0, device=ν.device))


def prox_L1(ν, κ):
    """soft-thresholding"""
    arg = torch.maximum(torch.abs(ν) - κ, torch.tensor(0.0, device=ν.device))
    return torch.sign(ν) * arg


def prox_isotropic_MCP(ν_list, λ, γ=20.0):
    """Proximal operator for isotropic Minimax Concave Penalty (MCP)

    Args:
        ν_list (list of torch.Tensor): List of tensors for each direction
        λ (float): Regularization parameter
        γ (float): MCP parameter that controls concavity, γ > 1

    Returns:
        list of torch.Tensor: Proximal operator result for each direction
    """
    # Compute joint magnitude across all directions
    squared_sum = sum(ν * ν for ν in ν_list)
    magnitude = torch.sqrt(
        squared_sum + 1e-10
    )  # small epsilon for numerical stability

    # Initialize output tensors
    out_list = []

    # Apply MCP thresholding to magnitude
    mask1 = magnitude <= γ * λ
    mask2 = magnitude > γ * λ

    # Scale factor for region 1 (|ν| ≤ γλ)
    scale1 = torch.zeros_like(magnitude)
    temp = magnitude - λ
    soft_thresh = torch.maximum(
        temp, torch.tensor(0.0, device=magnitude.device)
    )
    scale1[mask1] = soft_thresh[mask1] / ((1 - 1 / γ) * magnitude[mask1])

    # Scale factor for region 2 (|ν| > γλ)
    scale2 = torch.ones_like(magnitude)

    # Combine scales
    scale = torch.where(mask1, scale1, scale2)

    # Apply scaling to each direction
    for ν in ν_list:
        out_list.append(ν * scale)

    return torch.stack(out_list)


def prox_SCAD(ν, λ, a=2.5):
    """Proximal operator for Smoothly Clipped Absolute Deviation (SCAD) penalty

    Args:
        ν (torch.Tensor): Input tensor
        λ (float): Regularization parameter
        a (float): SCAD parameter, typically set to 3.7 as suggested in Fan and Li (2001)

    Returns:
        torch.Tensor: Proximal operator result
    """
    # Get absolute value and sign
    abs_ν = torch.abs(ν)
    sign_ν = torch.sign(ν)

    # Initialize output tensor
    out = torch.zeros_like(ν)

    # Region 1: |ν| ≤ λ
    mask1 = abs_ν <= λ
    out[mask1] = torch.sign(ν[mask1]) * torch.maximum(
        abs_ν[mask1] - λ, torch.tensor(0.0, device=ν.device)
    )

    # Region 2: λ < |ν| ≤ aλ
    mask2 = (abs_ν > λ) & (abs_ν <= a * λ)
    out[mask2] = ((a - 1) * ν[mask2] - sign_ν[mask2] * a * λ) / (a - 2)

    # Region 3: |ν| > aλ
    mask3 = abs_ν > a * λ
    out[mask3] = ν[mask3]

    return out


class DeconADMM:
    def __init__(self, data, otf, difference_operators, device="cuda"):
        self.device = device
        self.data = torch.as_tensor(data, device=self.device)
        self.shape = data.shape
        self.otf = torch.as_tensor(otf, device=self.device)
        self.L̃ = [
            torch.as_tensor(op, device=self.device)
            for op in difference_operators
        ]
        self.it = 0
        self.ONE = torch.ones_like(self.data)
        self.ZERO = torch.zeros_like(self.data)
        self.ndim = self.data.ndim

        # Setup FFT functions
        if self.ndim == 2:
            self.fft = lambda x: torch.fft.rfft2(x)
            self.ift = lambda x: torch.fft.irfft2(x, s=self.shape)
        elif self.ndim == 3:

            def _fft(x):
                if x.ndim > 3:
                    ft_axes = (1, 2, 3)
                    return torch.fft.rfftn(x, dim=ft_axes)
                else:
                    return torch.fft.rfftn(x)

            def _ift(x):
                if x.ndim > 3:
                    ft_axes = (1, 2, 3)
                    return torch.fft.irfftn(x, s=self.shape, dim=ft_axes)
                else:
                    return torch.fft.irfftn(x, s=self.shape, dim=(0, 1, 2))

            self.fft = _fft
            self.ift = _ift

        # Linear operators (Fourier)
        self.K = (
            self.otf,
            torch.tensor(1.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        ) + tuple(self.L̃)
        self.Nlinops = len(self.K)

        # Initialize variables
        self.x = torch.zeros_like(self.data)
        self.x̃ = None
        self.z = torch.zeros(
            (self.Nlinops, *self.data.shape), device=self.device
        )
        self.u = torch.zeros(
            (self.Nlinops, *self.data.shape), device=self.device
        )

        # Precompute inverse kernel
        otf2 = torch.conj(self.otf) * self.otf
        fHtH = torch.zeros_like(self.otf)
        for f in self.L̃:
            fHtH += torch.conj(f) * f
        self.invkernel = 2.0 + otf2 + fHtH

    def reset(self):
        self.it = 0
        self.z = torch.zeros(
            (self.Nlinops, *self.data.shape), device=self.device
        )
        self.u = torch.zeros(
            (self.Nlinops, *self.data.shape), device=self.device
        )
        self.x = torch.zeros_like(self.data)
        self.x̃ = None

    def step(self, ρ, ν, λ, regularization="L1", **kwargs):
        self.update_x()
        self.update_dual(ρ, ν, λ, penalty=regularization, **kwargs)
        self.it += 1
        RMSE = self.compute_RMSE()
        print(f"\rIteration {self.it:4d}, RMSE = {RMSE:10.4f}", end="")

    def compute_RMSE(self):
        Ax = self.ift(self.K[0] * self.x̃)
        err = Ax - self.data
        return torch.mean(torch.sqrt(err * err)).item()

    def update_x(self):
        ṽ = self.fft(self.z - self.u)
        KtV = sum([torch.conj(K_i) * ṽ_i for K_i, ṽ_i in zip(self.K, ṽ)])
        self.x̃ = KtV / self.invkernel

    def update_dual(self, ρ, ν, λ, penalty="L1", **kwargs):
        # Data fidelity term
        Kx = self.ift(self.K[0] * self.x̃)
        v = Kx + self.u[0, ...]
        self.z[0, ...] = prox_poisson(v, ρ, self.data)
        self.u[0, ...] += Kx - self.z[0, ...]

        # Positivity constraint
        Kx = self.ift(self.x̃)
        self.x = Kx
        v = Kx + self.u[1, ...]
        self.z[1, ...] = prox_indicator(v)
        self.u[1, ...] += Kx - self.z[1, ...]

        # Sparsity penalty
        v = self.x + self.u[2, ...]
        self.z[2, ...] = prox_isotropic_MCP(v, λ / ρ)
        self.u[2, ...] += Kx - self.z[2, ...]

        if penalty == "L1":
            for i in range(3, len(self.K)):
                Kx = self.ift(self.x̃ * self.K[i])
                v = Kx + self.u[i, ...]
                self.z[i, ...] = prox_L1(v, ν / ρ)
                self.u[i, ...] += Kx - self.z[i, ...]

        elif penalty == "L2":
            vlist = []
            for k in range(3, len(self.K)):
                Kx = self.ift(self.x̃ * self.K[k])
                v = Kx + self.u[k, ...]
                vlist.append(v)

            vnorm2 = torch.sqrt(sum([v * v for v in vlist]))
            vnorm2 = torch.maximum(
                vnorm2, torch.tensor(1e-8, device=self.device)
            )
            bsfact = torch.maximum(
                1.0 - (ν / ρ) / vnorm2, torch.tensor(0.0, device=self.device)
            )

            for k in range(3, len(self.K)):
                Kx = self.ift(self.x̃ * self.K[k])
                self.z[k, ...] = vlist[k - 3] * bsfact
                self.u[k, ...] += Kx - self.z[k, ...]

        elif penalty == "SCAD":
            for i in range(3, len(self.K)):
                Kx = self.ift(self.x̃ * self.K[i])
                v = Kx + self.u[i, ...]
                self.z[i, ...] = prox_SCAD(v, ν / ρ, **kwargs)
                self.u[i, ...] += Kx - self.z[i, ...]

        elif penalty == "MCP":
            for i in range(3, len(self.K)):
                Kx = self.ift(self.x̃ * self.K[i])
                v = Kx + self.u[i, ...]
                self.z[i, ...] = prox_isotropic_MCP(v, ν / ρ, **kwargs)
                self.u[i, ...] += Kx - self.z[i, ...]
