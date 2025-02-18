"""
This script demonstrates band extension and ringing artifact mitigation
by allowing the reconstruction to be solved in a much finer pixel spacing

It's done by formulating the forward / adjoint model using Fourier cropping and
padding, respectively.

Care must be taken when working with the transform because the pixel sizes are
not the same between 'hidden' space object and 'data'. Use the argument
'forward' in the FFT routines to get the correct intensities.

"""

# %%
import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np

matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

np.random.seed(0)  # For reproducibility

# %%
import numpy as np

# Create fine grid test object
N = 1000  # full resolution size
x = np.zeros(N)
x[400:420] = 50.0  # One sharp peak
x[430:480] = 20.0  # Another sharper peak

# Create Gaussian PSF in Fourier space
k = np.fft.rfftfreq(N)
sigma = 15
sigma_bandlimit = sigma / 5
H = np.exp(-2 * (np.pi * sigma * k) ** 2)
H_bandlimit = np.exp(-2 * (np.pi * sigma_bandlimit * k) ** 2)


# %%
# Forward model
def forward(x, H, n_coarse):
    # Full resolution convolution in Fourier space
    X = np.fft.rfft(x, norm="forward")
    X_blurred = X * H * H_bandlimit

    # Subsample Fourier coefficients for low resolution
    idx = np.r_[
        : n_coarse // 2, -n_coarse // 2 : 0
    ]  # Get coefficients around origin
    X_coarse = X_blurred[idx]

    return np.fft.irfft(X_coarse, n=n_coarse, norm="forward")


# Adjoint model
def adjoint(y, H, N):
    # Pad Fourier coefficients
    Y = np.fft.rfft(y, norm="forward")
    X_padded = np.zeros(N // 2 + 1, dtype=complex)
    X_padded[np.r_[: Y.size]] = Y

    # Full resolution convolution with conjugate
    return np.fft.irfft(
        X_padded * H.conj() * H_bandlimit.conj(), n=N, norm="forward"
    )


# %%

# Generate data
n_coarse = N // 4  # downsample by factor of 4
data = forward(x, H, n_coarse)
data = np.maximum(data, 1e-7)

# Add noise
data_noisy = np.random.poisson(data) + np.random.normal(0, 0.8, n_coarse)

x_data = np.linspace(0, 10, num=n_coarse)
x_model = np.linspace(0, 10, num=N)

plt.plot(x_data, data_noisy, "k-")
# %%
# Richardson-Lucy iteration
x_est = np.ones(N)
σ2 = 5.0

for i in range(100):
    pred = forward(x_est, H, n_coarse)
    ratio = (data_noisy + σ2) / (pred + σ2)
    # ratio = data_noisy / pred
    x_est = x_est * adjoint(ratio, H, N)

pred = forward(x_est, H, n_coarse)
x_out = np.fft.irfft(
    np.fft.rfft(x_est, norm="forward") * H_bandlimit, norm="forward"
)
fig, ax = plt.subplots()
ax.plot(x_data, data_noisy, "k.", ms=4)
ax.plot(x_data, pred, "m-", zorder=10)
ax.plot(x_model, x_out + 40, "r-", lw=2.0)
ax.plot(x_model, x + 120, "-", c="#708090")
ax.text(1, 5, "data /", color="k")
ax.text(2.2, 5, "model", color="m")
ax.text(1, 45, "deconvolved", color="r")
ax.text(1, 125, "object", color="#708090")
ax.yaxis.set_visible(False)
ax.spines["left"].set_visible(False)
# ax.set_xlim(2, 6)
# %%
