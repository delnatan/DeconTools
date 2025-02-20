import numpy as np
import torch

EPS = torch.finfo(torch.float32).eps
tEPS = torch.tensor(EPS, dtype=torch.float32)


def cg_solve(
    hvpfunc,
    d,
    reltol=1e-3,
    max_iter=30,
    mu_init=1e-4,
    verbose=False,
    eps=1e-6,
    **fkwargs,
):
    """
    Linear solver (conjugate gradient) without explicit matrix multiplication
    Solves (H + μI)x = d where H is implicitly defined by hvpfunc

    Uses adaptive regularization if system is badly-conditioned

    Args:
        hvpfunc (callable): evaluates hessian-vector product
        d (tensor): input vector, usually -gradient
        reltol (float): relative convergence tolerance
        max_iter (int): maximum iterations
        mu_init (float): initial regularization parameter
        verbose (bool): whether to print iteration progress
        eps (float): numerical stability threshold
        **fkwargs: additional keyword arguments passed to hvpfunc
    """
    x = torch.zeros_like(d)
    r = d.clone()
    p = r.clone()
    mu = max(mu_init, EPS)

    r_norm_sq = r.dot(r)
    initial_norm = torch.sqrt(r_norm_sq)

    residuals = []
    stagnation_counter = 0
    prev_rel_residual = float("inf")
    iter = 0

    while iter < max_iter:
        # Apply H + μI to p
        Ap = hvpfunc(p, **fkwargs).detach() + mu * p
        pAp = p.dot(Ap)

        if pAp < eps:
            mu *= 2.0
            if verbose:
                print(f"Warning: pAp < eps, increasing regularization to {mu}")
            # restart CG
            x = torch.zeros_like(d)
            r = d.clone()
            p = r.clone()
            r_norm_sq = r.dot(r)
            iter = 0
            continue

        alpha = r_norm_sq / pAp

        if not torch.isfinite(alpha) or alpha > 1e6:
            mu *= 2.0
            if verbose:
                print(
                    f"Warning: alpha too large, increasing regularization to {mu}"
                )
            # restart CG
            x = torch.zeros_like(d)
            r = d.clone()
            p = r.clone()
            r_norm_sq = r.dot(r)
            iter = 0
            continue

        # If we get here, we have a valid iteration
        iter += 1

        x += alpha * p
        r -= alpha * Ap

        r_next_norm_sq = r.dot(r)
        rel_residual = torch.sqrt(r_next_norm_sq) / initial_norm
        residuals.append(rel_residual.item())

        if verbose:
            print(
                f"CG iteration {iter:3d}: |r| = {rel_residual:12.5E}, |μ| = {mu:12.5E}"
            )

        if abs(prev_rel_residual - rel_residual.item()) < eps:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= 3:
            mu *= 2.0
            if verbose:
                print("Warning: iteration stagnated")
            # restart CG
            x = torch.zeros_like(d)
            r = d.clone()
            p = r.clone()
            r_norm_sq = r.dot(r)
            stagnation_counter = 0
            iter = 0
            continue

        if rel_residual < reltol:
            break

        beta = r_next_norm_sq / r_norm_sq

        if not torch.isfinite(beta):
            if verbose:
                print("Warning: beta not finite!")
            # restart CG
            x = torch.zeros_like(d)
            r = d.clone()
            p = r.clone()
            r_norm_sq = r.dot(r)
            iter = 0
            continue

        p = r + beta * p
        r_norm_sq = r_next_norm_sq
        prev_rel_residual = rel_residual.item()

    return x, residuals, mu


def js_cg_solve(
    hvpfunc,
    d,
    fkwargs=None,
    beta=None,
    reltol=1e-4,
    max_iter=25,
    verbose=False,
):
    """John Skilling's conjugate gradient

    modified to solve Ax = b without R

    The idea is to enable access to the symmetric tridiagonal matrix
    to do what Skilling propose by doing a two-pass CG iteration:
    1) Obtain T, do SVD to obtain the subspace vectors and singular values
    2) Re-run CG with same parameters as 1 (e.g. number of iterations) to get
    orthonormal bases and evaluate 'simulated' function by augmenting the
    singular values.

    Args:
        hvpfunc(callable): function that evaluates the hessian-vector product
        d (tensor): input tensor
        fkwargs (dict): dictionary of parameters passed to `hvpfunc`
        reltol (float): tolerance for relative convergence
        beta (float): tikhonov's regularization factor

    """
    if fkwargs is None:
        fkwargs = {}

    g = d.clone()
    g0 = torch.sqrt(torch.dot(g, g))
    q = 0.0

    x = torch.zeros_like(d)
    h = torch.zeros_like(d)
    del2_prev = 0.0
    gam_prev = 0.0

    # alphas (diagonals)
    alphas = []
    # betas (off-diagonals)
    betas = []

    for k in range(max_iter):
        gam2 = torch.dot(g, g)
        gam = torch.sqrt(gam2)

        # check convergence and exit if met
        rel_residual = gam / g0

        if rel_residual < reltol:
            break

        h += g / gam2

        if beta is not None:
            Ah = hvpfunc(h, **fkwargs) + beta * h
        else:
            Ah = hvpfunc(h, **fkwargs)

        del2 = torch.dot(h, Ah)

        q += 1 / del2

        if verbose:
            print(
                f"  iter. {k + 1:2d}: |r| = {rel_residual:11.5E}, q = {q:11.5E}"
            )

        # compute diagonal element
        alphas.append(gam2 * (del2_prev + del2))

        if k > 0:
            betas.append(-del2_prev * gam_prev * gam)

        g -= Ah / del2
        x += h / del2

        gam_prev = gam
        del2_prev = del2

    return x, torch.tensor(alphas), torch.tensor(betas)


def solve_normalized_cubic(a, b, d):
    """solve cubic equation where c=1

    a := 3 * g^t.A.A.g
    b := 2 * g^t.A.g
    c := g^t.g
    d := (rate * r_0^2)

    the solution is 1/β where

    g^t.(A + β*I)^(-2).g = rate * r_0^2

    here g = [√f]·∇Q

    the cubic equation was derived from using the Neumann series expansion
    for matrix inverse.

    """
    c = 1
    scale = np.cbrt(abs(d) / a)
    # scale the variables
    b /= scale
    c /= scale * scale
    d /= scale * scale * scale

    p = (3 * a * c - b * b) / (3 * a * a)
    q = (2 * b * b * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a)

    discr = (q * q) / 4 + (p * p * p) / 27

    if discr > 0:
        sqrt_discr = np.sqrt(discr)
        t = np.cbrt(-q / 2 + sqrt_discr) + np.cbrt(-q / 2 - sqrt_discr)
        x = t - b / (3 * a)
        x *= scale
    else:
        A = 2 * np.sqrt(-p / 3)
        theta = np.arccos(-q / (2 * np.sqrt(-(p * p * p) / 27))) / 3
        x1 = A * np.cos(theta)
        x2 = A * np.cos(theta + 2 * np.pi / 3)
        x3 = A * np.cos(theta + 4 * np.pi / 3)
        x = np.array([x1, x2, x3]) - b / (3 * a)
        x *= scale
    return x
