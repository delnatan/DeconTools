# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

np.random.seed(0)  # For reproducibility

# %%
import numpy as np

# Create fine grid test object
N = 1000  # full resolution size
x = np.zeros(N)
x[400:420] = 1.0  # One sharp peak
x[430:450] = 2.0  # Another sharper peak

# Create Gaussian PSF in Fourier space
k = np.fft.fftfreq(N)
sigma = 12
sigma_bandlimit = sigma / 5
H = np.exp(-2 * (np.pi * sigma * k) ** 2)
H_bandlimit = np.exp(-2 * (np.pi * sigma_bandlimit * k) ** 2)


# %%
# Forward model
def forward(x, H, n_coarse):
    # Full resolution convolution in Fourier space
    X = np.fft.fft(x)
    X_blurred = X * H * H_bandlimit

    # Subsample Fourier coefficients for low resolution
    idx = np.r_[
        : n_coarse // 2, -n_coarse // 2 : 0
    ]  # Get coefficients around origin
    X_coarse = X_blurred[idx]

    return np.fft.ifft(X_coarse).real


# Adjoint model
def adjoint(y, H, N):
    # Pad Fourier coefficients
    n_coarse = len(y)
    Y = np.fft.fft(y)
    X_padded = np.zeros(N, dtype=complex)
    X_padded[np.r_[: n_coarse // 2, -n_coarse // 2 : 0]] = Y

    # Full resolution convolution with conjugate
    return np.fft.ifft(X_padded * H.conj() * H_bandlimit.conj()).real


# %%

# Generate data
n_coarse = N // 4  # downsample by factor of 4
data = forward(x, H, n_coarse)
data = np.maximum(data, 1e-7)

# Add noise
data_noisy = np.random.poisson(data * 40) / 100 + np.random.normal(
    0, 0.02, n_coarse
)

plt.plot(data_noisy, "k-")
# %%
# Richardson-Lucy iteration
x_est = np.ones(N)
omega = 0.1
for i in range(100):
    pred = forward(x_est, H, n_coarse)
    ratio = np.zeros_like(data_noisy)
    mask = pred > 0
    ratio[mask] = data_noisy[mask] / pred[mask]
    x_est = x_est * adjoint(ratio, H, N)
    x_est = omega * np.maximum(x_est, 0.0) + (1 - omega) * x_est

plt.plot(x_est, "r-")
plt.plot(x * 0.25, "k-", lw=0.5)
# %%

plt.plot(data_noisy, "k.-")
n
