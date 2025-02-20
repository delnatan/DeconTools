"""
this little toy study is meant to check our intuition on the effect of
non-uniform baselines (often lower-frequency) on the Richardson-Lucy
algorithm.

To some extent, a 'flat' background subtraction can mitigate some of the
ringing effect. But for a more 'correct' peak location reconstruction,
a more sophisticated method is needed to estimate the background.


"""

# %%
import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
import torch

import toy_problem as tp

matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%
np.random.seed(18)

N = 600
x = np.arange(-60, N + 60)
bg = tp.make_nonuniform_background(
    N + 2 * 60, magnitude=8, correlation_length=120
)

signal = sum(
    [
        tp.lorentzian(x, 200, 12, 250),
        tp.lorentzian(x, 250, 8, 120),
        tp.lorentzian(x, 400, 5.0, 300),
        tp.lorentzian(x, 430, 4.0, 220),
    ]
)

# add 'bump'
clean_signal = signal + bg
plt.plot(clean_signal)
plt.plot(bg, "r--")

torch_signal = torch.from_numpy(clean_signal.astype(np.float32))

# %% generate data
zoom = 3

M = tp.ToyModel(200, 3, 20, 7.5, 10.0)

x = torch.rand(M.N) * 10  # x
y = torch.rand(M.M - M.pad * 2) * 10  # y

Ax = M.forward(x)
Aty = M.adjoint(y)

lhs = torch.dot(y, Ax)
rhs = torch.dot(x, Aty)

print(f"Transpose check")
print(f"lhs = {lhs:15.5E}, rhs = {rhs:15.5E}")
print(f"|lhs - rhs| = {torch.abs(lhs - rhs):15.5E}")
# %%
b_clean = M.forward(torch_signal)
b = torch.poisson(b_clean)
tbg = M.forward(torch.from_numpy(bg))
b2 = torch.clamp(b - 390, 1e-6)
# %%
xest = torch.ones(M.N)
ones = torch.ones(200)
hnorm = M.adjoint(ones)
reps = 1000

for k in range(8000):
    model = M.forward(xest)
    ratio = (b2 + reps) / (model + reps)
    update = M.adjoint(ratio)
    xest *= update
    xest = torch.where(hnorm > 1e-2, xest / hnorm, xest)

fig, ax = plt.subplots(ncols=2, figsize=(10, 3.75))
ax[0].plot(b2, "k.")
ax[0].plot(model, "r-")
ax[0].plot(tbg, "--", c="0.5")
ax[1].plot(clean_signal[60:-60], "-", lw=1.2, c="purple")
ax[1].plot(xest[60:-60], "r-", lw=0.95)
