from typing import List, Optional, Tuple

import numpy as np
import torch


class LinearOperator:
    """FFT-based linear operator that works with either numpy arrays or torch
    tensors

    """

    def __init__(self, kernel, shape, in_fourier_space=False):
        """
        Args:
            kernel: numpy array or torch tensor
            shape: desired output shape
            in_fourier_space: if True, kernel is assumed to be in Fourier space
            already
        """

        self.shape = shape

        # check kernel dimensionality and input shape
        if len(kernel.shape) != len(shape):
            raise ValueError(
                f"Kernel dimensionality {len(kernel.shape)} "
                f"doesn't match target shape dimensionality {len(shape)}"
            )

        # check fourier kernel for matching shape
        if in_fourier_space and kernel.shape != shape:
            raise ValueError("Fourier-space kernel must match target shape")

        # Detect backend from input type
        if hasattr(kernel, "device"):  # torch tensor
            import torch
            import torch.fft

            self.fft = torch.fft
            self.kernel = kernel.to(torch.float32)
        else:  # numpy array
            from scipy import fft

            self.fft = fft
            self.kernel = kernel.astype(np.float32)

        # Pad kernel if needed
        if self.kernel.shape != shape:
            self.kernel = self._pad_kernel(self.kernel, shape)

        if not in_fourier_space:
            # Precompute FFT of kernel
            self.ft_kernel = self.fft.rfftn(self.kernel)
        else:
            self.ft_kernel = kernel

    def dot(self, x, invert=True):
        """Forward operation A*x"""
        x_fft = self.fft.rfftn(x, s=self.shape)
        if invert:
            return self.fft.irfftn(self.ft_kernel * x_fft, s=self.shape)
        else:
            return self.ft_kernel * x_fft

    def adjoint(self, x, invert=True):
        """Adjoint operation A^T*x"""
        x_fft = self.fft.rfftn(x)

        if invert:
            return self.fft.irfftn(x_fft * self.ft_kernel.conj(), s=self.shape)
        else:
            return self.ft_kernel.conj() * x_fft

    def return_self_adjoint(self):
        """returns self-adjoint kernel in Fourier-space AᵗA"""
        return self.ft_kernel.conj() * self.ft_kernel

    def invdot(self, x):
        """Inverse operation A^{-1}*x"""
        x_fft = self.fft.rfftn(x)
        return self.fft.irfftn(x_fft / self.ft_kernel, s=self.shape)

    def __call__(self, x):
        return self.dot(x)

    def _pad_kernel(self, kernel, target_shape: Tuple[int]) -> np.ndarray:
        """Zero-pad a kernel that has its origin at index 0 to target shape,
        maintaining FFT-compatible positioning (origin at 0, negative indices
        wrapped to end)

        """

        is_torch = hasattr(kernel, "device")

        if is_torch:
            kernel_np = kernel.cpu().numpy()
        else:
            kernel_np = kernel

        slices = []

        for s in kernel.shape:
            w = s // 2
            indices = np.r_[0 : w + (s % 2 == 0), -w:0]
            slices.append(indices)

        padded = np.zeros(target_shape, dtype=kernel_np.dtype)
        padded[np.ix_(*slices)] = kernel_np

        if is_torch:
            import torch

            padded = torch.from_numpy(padded).to(kernel.device)

        return padded


class CompositeLinearOperator:
    def __init__(
        self,
        operators: List[LinearOperator],
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            operators: List of LinearOperator instances
            weights: Optional weights for each operator (e.g. sqrt(2) for mixed
            derivatives)
        """
        self.operators = operators
        self.weights = (
            weights if weights is not None else [1.0] * len(operators)
        )

        # Verify all operators have same shape
        self.shape = operators[0].shape
        for op in operators:
            assert op.shape == self.shape

    def dot(self, x, invert=True) -> List:
        """Compute all operations D*x and return as a list"""
        return [
            w * op.dot(x, invert=invert)
            for w, op in zip(self.weights, self.operators)
        ]

    def adjoint(self, x: list, invert=False):
        """adjoint of finite difference operators"""
        return sum(
            w**2 * op.adjoint(_x, invert=invert)
            for w, op, _x in zip(self.weights, self.operators, x)
        )

    def return_self_adjoint(self):
        return sum(
            w**2 * op.return_self_adjoint()
            for w, op in zip(self.weights, self.operators)
        )

    def __call__(self, x) -> List:
        return self.dot(x)

    def L2norm_squared(self, x) -> float:
        """Compute 0.5 * ||D*x||_2^2"""
        results = self.dot(x)
        if hasattr(results[0], "device"):  # torch tensor
            return float(sum((r**2).sum() for r in results))
        return float(0.5 * sum(np.sum(r**2) for r in results))

    def L2norm_gradient(self, x):
        """Compute gradient of 0.5*||D*x||_2^2 which is D^T(D*x)"""
        # First compute D*x
        Dx = self.dot(x)

        # Initialize gradient with same type as input
        if hasattr(x, "device"):  # torch tensor
            import torch

            grad = torch.zeros_like(x)
        else:
            grad = np.zeros_like(x)

        # Then compute D^T(D*x) for each operator and sum
        for w, op, dx in zip(self.weights, self.operators, Dx):
            grad += w * op.adjoint(dx)

        return grad


class FourierCropper:
    def __init__(self, input_shape: Tuple[int], crop_shape: Tuple[int]):
        """class for making convenient crop/padding operation

        Creates a Fourier cropping operator C: ℝᴺ → ℝᴹ
        Where N is the number of elements in the 'object' space
        and M is the number of elements in undersampled 'object' space

        C = FourierCropper((256, 256), (128, 128))

        # Fourier cropping (or 'binning')
        # crop fourier array x from 256, 256 to 128,128
        Cx = C.dot(x)

        # Fourier padding (or 'interpolation')
        # pads Cx back to input size
        CtCx = C.adjoint(Cx)


        """
        self.input_shape = input_shape
        self.crop_shape = crop_shape

        # Generate indices for all dimensions except last
        self.indices = []
        for in_size, crop_size in zip(input_shape[:-1], crop_shape[:-1]):
            first_half = np.arange(0, (crop_size + 1) // 2)
            second_half = np.arange(in_size - crop_size // 2, in_size)
            self.indices.append(np.r_[first_half, second_half])

        # Last dimension only needs first half due to rfftn
        self.indices.append(np.arange(crop_shape[-1] // 2 + 1))

        # Create indexing tuple
        self.idx_tuple = np.ix_(*self.indices)

        # For PyTorch
        self.torch_indices = [torch.from_numpy(idx) for idx in self.indices]

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x_f = torch.fft.rfftn(x)
            return torch.fft.irfftn(
                x_f[torch.meshgrid(*self.torch_indices, indexing="ij")],
                s=self.crop_shape,
            )
        else:
            x_f = np.fft.rfftn(x)
            return np.fft.irfftn(x_f[self.idx_tuple], s=self.crop_shape)

    def adjoint(self, x):
        if isinstance(x, torch.Tensor):
            x_f = torch.fft.rfftn(x)
            x_padded = torch.zeros(
                (*self.input_shape[:-1], self.input_shape[-1] // 2 + 1),
                dtype=torch.complex64,
                device=x.device,
            )
            x_padded[torch.meshgrid(*self.torch_indices, indexing="ij")] = x_f
            return torch.fft.irfftn(x_padded, s=self.input_shape)
        else:
            x_f = np.fft.rfftn(x)
            x_padded = np.zeros(
                (*self.input_shape[:-1], self.input_shape[-1] // 2 + 1),
                dtype=np.complex64,
            )
            x_padded[self.idx_tuple] = x_f
            return np.fft.irfftn(x_padded, s=self.input_shape)
