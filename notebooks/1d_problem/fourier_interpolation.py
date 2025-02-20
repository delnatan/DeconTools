# %%
import matplotlib.pyplot as plt
import torch

# %%
m = 20
zoom = 2.5
n = int(zoom * m)

# x = [1 1 1 1 ...]
vec_x = torch.ones(m, dtype=torch.float32)

# its FT is just a delta function at origin ∑x_i = m
vec_X = torch.fft.rfft(vec_x)

# compute the FT{x} indices
X_idx = (torch.fft.rfftfreq(m) * m).type(torch.int64)

# define larger complex array
n_nyquist = n // 2 + (n % 2 == 0)

# zero pad by filling in Fourier coefficients into larger array
ft_Y = torch.zeros(n_nyquist, dtype=torch.complex64)
ft_Y[X_idx] = vec_X[X_idx]

vec_y = torch.fft.irfft(ft_Y, n=n)

# without accounting for binning the output is divided by 1/n
# so the output is actually going to be x * (m / n)

# this has the important property that it would preserve the total
# 'energy' of the input/output image
torch.allclose(vec_x.sum(), vec_y.sum())

vec_y_scaled = torch.fft.irfft(ft_Y * (n / m), n=n)

# now to keep the intensities at the same scale as input vector
# the coefficients need to be multiplied by (1 / m) = (1 / n) * (n / m)
# we need to scale it back by the effective 'zoom' scale, (n / m)
