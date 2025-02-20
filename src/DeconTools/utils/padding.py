import torch
import torch.nn.functional as F


def torch_ix_(*args):
    """
    Mimics numpy.ix_ in PyTorch, returning a tuple of index arrays that can be
    used for open-mesh (broadcasted) indexing.

    Each input tensor is reshaped to have singleton dimensions so that when used
    together, they form an open mesh.

    Example:
        a = torch.tensor([0, 1, 2])
        b = torch.tensor([10, 20])
        # Instead of doing torch.meshgrid(a, b) which creates a full grid,
        # torch_ix_(a, b) returns (a.reshape(3, 1), b.reshape(1, 2))
    """
    return tuple(
        arg.reshape(-1, *[1] * (len(args) - i - 1))
        for i, arg in enumerate(args)
    )


def calculate_frequency_indices(n: int) -> torch.Tensor:
    positive_indices = torch.arange(0, n // 2 + 1)
    negative_indices = torch.arange(-(n - (n // 2 + 1)), 0)
    return torch.cat((positive_indices, negative_indices))


class SpatialPad:
    def __init__(
        self, original_shape: tuple[int, ...], padded_shape: tuple[int, ...]
    ):
        self.original_shape = original_shape
        self.padded_shape = padded_shape

        if len(original_shape) != len(padded_shape):
            raise ValueError(
                "original_shape and padded_shape must have the same number"
                " of dimensions"
            )

        self.original_shape = tuple(original_shape)
        self.padded_shape = tuple(padded_shape)
        self.ndim = len(original_shape)

        self.pad_amounts = []
        for orig, pad in zip(self.original_shape, self.padded_shape):
            if pad < orig:
                raise ValueError(
                    "Each dimension in padded_shape must be >= original_shape."
                )
            total_pad = pad - orig
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            self.pad_amounts.append((pad_before, pad_after))

    def pad(self, x):
        pad_list = []
        for pad_before, pad_after in reversed(self.pad_amounts):
            pad_list.extend([pad_before, pad_after])
        return F.pad(x, pad_list)

    def unpad(self, x):
        # Build slices for the spatial dimensions.
        slices = [slice(None)] * (x.ndim - self.ndim)
        for (pad_before, _), orig in zip(
            self.pad_amounts, self.original_shape
        ):
            slices.append(slice(pad_before, pad_before + orig))
        return x[tuple(slices)]


class OriginPad:
    def __init__(
        self, original_shape: tuple[int, ...], padded_shape: tuple[int, ...]
    ):
        self.original_shape = original_shape
        self.padded_shape = padded_shape
        self.ndim = len(original_shape)

        if len(original_shape) != len(padded_shape):
            raise ValueError(
                "original_shape and padded_shape must have the same number of"
                " dimensions"
            )

        # compute RFFT shapes
        self.original_rfft_shape = self.original_shape[:-1] + (
            self.original_shape[-1] // 2 + 1,
        )
        self.padded_rfft_shape = self.padded_shape[:-1] + (
            self.padded_shape[-1] // 2 + 1,
        )

        # compute frequency indices
        valid_freq_index = tuple(
            calculate_frequency_indices(n)
            if i < self.ndim
            else torch.arange(0, n // 2 + 1)
            for i, n in enumerate(self.original_shape, start=1)
        )

        valid_index = tuple(
            calculate_frequency_indices(n) for n in self.original_shape
        )

        self.valid_freq_index = torch_ix_(*valid_freq_index)
        self.valid_index = torch_ix_(*valid_index)

    def pad(self, x: torch.Tensor, frequency=True):
        new_shape = self.padded_rfft_shape if frequency else self.padded_shape
        valid = self.valid_freq_index if frequency else self.valid_index
        valid_device = tuple(idx.to(x.device) for idx in valid)
        padded = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
        padded[valid_device] = x[valid_device]
        return padded

    def unpad(self, x: torch.Tensor, frequency=True):
        valid = self.valid_freq_index if frequency else self.valid_index
        valid_device = tuple(idx.to(x.device) for idx in valid)
        return x[valid_device]
