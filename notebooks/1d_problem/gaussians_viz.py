"""
Visualize and check to see if the Fourier expression for the Gaussian
filter is correct by doing an inverse FFT of the output coefficients
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import toy_problem as tp

# %%
n = 250
rg = tp.gaussian_kernel_1d(n, 10, fourier=False)

fG = tp.gaussian_kernel_1d(n, 10, fourier=True)

rginv = np.fft.irfft(fG, n=n)

# %%
fig, ax = plt.subplots()

ax.plot(rg / rg[0], "k.")
ax.plot(fG, "m--")
ax.plot(rginv / rginv[0], "r-")
