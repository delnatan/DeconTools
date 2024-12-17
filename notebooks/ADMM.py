import numpy as np
from scipy.special import lambertw


def prox_poisson(ν, ρ, b):
    """

    Args:
        ν (nd-array): input
    """
    arg = (1 - ρ * ν) / ρ / 2.0
    arg2 = np.maximum(arg * arg + b / ρ, 0.0)
    return -arg + np.sqrt(arg2)


def prox_indicator(ν):
    return np.maximum(ν, 0.0)


def prox_L1(ν, κ):
    """soft-thresholding"""
    arg = np.maximum(np.abs(ν) - κ, 0.0)
    return np.sign(ν) * arg


def prox_entropy(ν, α):
    arg = np.exp(ν / α - 1.0) / α
    return α * lambertw(arg)


def prox_entropy_L1(ν, λ):
    """dual-regularized entroy + L1

    See
    https://hal.archives-ouvertes.fr/hal-01421741/document
    """

    arg = np.exp(ν / λ - 1.0) / λ

    return λ * np.real(lambertw(arg))


class DeconADMM:
    """Linear-inverse problem with ADMM

    Args:
        data (nd-array):
            input data
        otf (complex nd-array):
            Fourier transform of the OTF
        difference_operators (list of complex nd-array):
            list of finite difference operators as fourier-filters

    """

    def __init__(self, data, otf, difference_operators):
        self.data = data
        self.shape = data.shape
        self.otf = otf
        self.L̃ = difference_operators
        self.it = 0
        self.ONE = np.ones_like(self.data)
        self.ZERO = np.zeros_like(self.data)
        self.ndim = self.data.ndim

        if self.ndim == 2:
            self.fft = np.fft.rfft2
            self.ift = np.fft.irfft2
        elif self.ndim == 3:

            def _fft(x):
                if x.ndim > 3:
                    ft_axes = (1, 2, 3)
                    return np.fft.rfftn(x, axes=ft_axes)
                else:
                    return np.fft.rfftn(x)

            def _ift(x):
                if x.ndim > 3:
                    ft_axes = (1, 2, 3)
                    return np.fft.irfftn(x, axes=ft_axes, s=self.shape)
                else:
                    return np.fft.irfftn(x, axes=(0, 1, 2), s=self.shape)

            self.fft = _fft
            self.ift = _ift

        # linear operators (Fourier)
        self.K = (self.otf, 1.0, 1.0) + self.L̃

        self.Nlinops = len(self.K)

        # primal variable
        self.x = np.zeros_like(self.data)
        self.x̃ = None

        # dual variable
        self.z = np.zeros((self.Nlinops, *self.data.shape))

        # dual residual
        self.u = np.zeros((self.Nlinops, *self.data.shape))

        otf2 = np.conj(self.otf) * self.otf

        fHtH = np.zeros_like(self.otf)

        for f in self.L̃:
            fHtH += np.conj(f) * f

        self.invkernel = 2.0 + otf2 + fHtH

    def reset(self):
        # dual variable
        self.it = 0
        self.z = np.zeros((self.Nlinops, *self.data.shape))
        self.u = np.zeros((self.Nlinops, *self.data.shape))
        self.x = np.zeros_like(self.data)
        self.x̃ = None

    def step(self, ρ, ν, λ, regularization="L1"):
        """single step of the ADMM update

        Args:
            ρ (float): ADMM damping factor to prevent division by zero.
            Usually 1e-3 to 1e-6.
            ν (float): regularization factor for norm of Hessian
            λ (float): sparsity enforcement factor

        """
        self.update_x()
        self.update_dual(ρ, ν, λ, penalty=regularization)
        self.it += 1
        RMSE = self.compute_RMSE()
        print("\rIteration %4d, RMSE = %10.4f" % (self.it, RMSE), end="")

    def compute_RMSE(self):
        Ax = self.ift(self.K[0] * self.x̃)
        err = Ax - self.data
        return np.mean(np.sqrt(err * err))

    def update_x(self):
        ṽ = self.fft(self.z - self.u)
        # K^t.v
        KtV = sum([np.conj(K_i) * ṽ_i for K_i, ṽ_i in zip(self.K, ṽ)])
        # keep solution in fourier space
        self.x̃ = KtV / self.invkernel

    def update_dual(self, ρ, ν, λ, penalty="L1"):
        # z1 & u1 (data-fidelity term)
        Kx = self.ift(self.K[0] * self.x̃)
        v = Kx + self.u[0, ...]
        self.z[0, ...] = prox_poisson(v, ρ, self.data)
        self.u[0, ...] += Kx - self.z[0, ...]

        # z2 & u2 (positivity constraint term)
        Kx = self.ift(self.x̃)
        self.x = Kx
        v = Kx + self.u[1, ...]
        self.z[1, ...] = prox_indicator(v)
        self.u[1, ...] += Kx - self.z[1, ...]

        # z3 & u3 (sparsity penalty term)
        v = self.x + self.u[2, ...]
        self.z[2, ...] = prox_L1(v, λ / ρ)
        self.z[2, ...] += Kx - self.z[2, ...]

        if penalty == "L1":
            for i in range(3, len(self.K)):
                Kx = self.ift(self.x̃ * self.K[i])
                self.z[i, ...] = prox_L1(v, ν / ρ)
                self.u[i, ...] += Kx - self.z[i, ...]

        if penalty == "entropy":
            for i in range(3, len(self.K)):
                Kx = self.ift(self.x̃ * self.K[i])
                v = Kx * Kx + self.u[i, ...]
                # v = Kx + self.u[i, ...]
                self.z[i, ...] = prox_entropy_L1(v, ν / ρ)
                self.u[i, ...] += Kx - self.z[i, ...]

        elif penalty == "L2":
            vlist = []
            for k in range(3, len(self.K)):
                Kx = self.ift(self.x̃ * self.K[k])
                v = Kx + self.u[k, ...]
                vlist.append(v)

            vnorm2 = np.sqrt(sum([v * v for v in vlist]))
            vnorm2 = np.maximum(vnorm2, 1e-8)
            bsfact = np.maximum(1.0 - (ν / ρ) / vnorm2, 0.0)

            for k in range(3, len(self.K)):
                Kx = self.ift(self.x̃ * self.K[k])
                self.z[k, ...] = vlist[k - 3] * bsfact
                self.u[k, ...] += Kx - self.z[k, ...]
