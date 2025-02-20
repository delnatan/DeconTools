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
from DeconTools.optim.solvers import js_cg_solve

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

# %% generate data by using our forward model
b_clean = M.forward(torch_signal)

# quantize simulated data
b = torch.poisson(b_clean)

# convert numpy baseline into torch
tbg = M.forward(torch.from_numpy(bg))

# b with 'incorrect' flat background
bflat = torch.clamp(b - 320, 1e-6)

plt.plot(bflat)

# %%
xest = torch.ones(M.N)
ones = torch.ones(200)
hnorm = M.adjoint(ones)
reps = 1000

for k in range(200):
    model = M.forward(xest)
    ratio = (bflat + reps) / (model + reps)
    update = M.adjoint(ratio)
    xest *= update
    xest = torch.where(hnorm > 1e-2, xest / hnorm, xest)

fig, ax = plt.subplots(ncols=2, figsize=(10, 3.75))
ax[0].plot(bflat, "k.")
ax[0].plot(model, "r-")
ax[1].plot(signal[60:-60], "-", lw=1.2, c="purple")
ax[1].plot(xest[60:-60], "r-", lw=0.95)


# %% MEM iteration
xmem = torch.ones(M.N)
memprior = torch.ones(M.N)


def J_and_gradJ(f, D, α, m):
    model = M.forward(f)
    residual = D - model
    data_loss = torch.sum(residual * residual) / 2.0
    entropy_loss = torch.sum(f * torch.log(f / m) + m - f)

    fval = data_loss + α * entropy_loss

    # evaluate gradient also
    gradL = -M.adjoint(residual)
    gradS = α * torch.log(f / m)

    return fval, gradL + gradS


def hvpfunc(v, metric):
    x = metric * v
    x = M.forward(x)
    x = M.adjoint(x)
    return metric * x


# %%
alpha = 1e4
alpha_min = 1e-2

m = torch.ones(M.N)

tol = 1e-4

# %% MEM alpha anneal loop
while alpha > alpha_min:
    converged = False

    print(f"### Alpha iteration = {alpha:15.5E} ###")

    while not converged:
        # Compute gradient and Hessian
        fval, g = J_and_gradJ(xmem, bflat, alpha, m)
        metric = torch.sqrt(xmem)

        beta = max(beta_min, alpha)  # Start beta at least as large as alpha
        r0 = sum(xmem)

        while True:  # Beta adjustment loop
            dx, _, _ = js_cg_solve(
                hvpfunc,
                -metric * g,
                fkwargs={"metric": metric},
                verbose=True,
                beta=beta,
            )

            r = sum(dx**2 / xmem)

            if r <= r0:
                break
            beta *= 1.618  # Or use more sophisticated update

        if r <= r0:
            xmem_new = xmem + dx
            relative_change = torch.linalg.norm(
                xmem_new - xmem
            ) / np.linalg.norm(xmem)
            # Check convergence criteria
            if relative_change < tol:
                converged = True

    # Decrease alpha for next outer iteration
    alpha *= 0.618  # Or use different factor


# %%

r0sq = torch.sum(xmem)
r = float("inf")

if r < r0sq:
    # decrease beta
    beta *= 0.5

while r > r0sq:
    df, _, _ = js_cg_solve(
        hvpfunc,
        -metric * g,
        fkwargs={"metric": metric},
        verbose=True,
        beta=beta,
    )

    df = metric * df

    # quantify length
    r = torch.sum(df * df / xmem)

    print(f"Beta control")
    print(f"beta = {beta:8.4f}, |r|^2 = {r:15.5E} < r0^2 = {r0sq:15.5E}")

    # increase beta
    beta *= 1.618


# %%
xmem += df
