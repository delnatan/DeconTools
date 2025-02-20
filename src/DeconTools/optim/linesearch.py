"""
12/30/2024: implemented quadratic line search from Markham & Conchello's code.
"""


def quadratic_line_search(
    f, x, direction, initial_step=0.5, max_steps=10.0, *args, **kwargs
):
    """
    Performs quadratic line search to find optimal step size.

    Args:
        obj_fn: Callable that computes objective function value
        x: Current parameters (torch.Tensor)
        direction: Search direction (torch.Tensor)
        initial_step: Initial step size (float)
        max_steps: Maximum number of step size adjustments

    Returns:
        optimal_step: Optimal step size (float)
    """
    h2 = initial_step

    # Get initial objective value
    f1 = f(x, *args, **kwargs)

    # Try initial step
    x2 = x + h2 * direction
    f2 = f(x2, *args, **kwargs)
    steps = 0
    while f2 >= f1 and steps < max_steps:
        h2 = h2 / 2
        x2 = x + h2 * direction
        f2 = f(x2, *args, **kwargs)
        steps += 1

    if f2 >= f1:
        return 0.0  # No improvement found

    # Try larger step to bracket minimum
    h3 = 2 * h2
    x3 = x + h3 * direction
    f3 = f(x3, *args, **kwargs)

    # Keep doubling until we bracket the minimum
    while f3 < f2 and steps < max_steps:
        h2 = h3
        f2 = f3
        h3 = 2 * h3
        x3 = x + h3 * direction
        f3 = f(x3, *args, **kwargs)
        steps += 1

    # Quadratic interpolation to find minimum
    # f(h) ≈ ah² + bh + c
    # Using three points: (0,f1), (h2,f2), (h3,f3)
    denom = (h2 - h3) * h2 * h3
    if abs(denom) < 1e-10:
        return h2  # Avoid division by zero

    a = (h3 * (f2 - f1) - h2 * (f3 - f1)) / denom
    b = (-h3 * h3 * (f2 - f1) + h2 * h2 * (f3 - f1)) / denom

    if a > 0:  # Check if we found a minimum
        optimal_step = -b / (2 * a)
        # Keep step size within bracket
        optimal_step = min(max(optimal_step, min(h2, h3)), max(h2, h3))
    else:
        optimal_step = h2  # Use best known point if no minimum found

    return optimal_step
