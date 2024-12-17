from typing import List, Tuple, Union

import numpy as np
import torch


def _process_input(
    x: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
) -> Tuple:
    """
        Process input to determine type and convert list to consistent format if
        needed.  Returns tuple of (processed_input, is_torch, is_list,
        original_type)
    y"""
    is_torch = False
    is_list = isinstance(x, list)
    original_type = type(x)

    if is_list:
        if len(x) == 0:
            raise ValueError("Empty list provided")
        first_elem = x[0]
        is_torch = isinstance(first_elem, torch.Tensor)
        if is_torch:
            return (
                [
                    t if isinstance(t, torch.Tensor) else torch.from_numpy(t)
                    for t in x
                ],
                True,
                True,
                original_type,
            )
        else:
            return (
                [t if isinstance(t, np.ndarray) else t.numpy() for t in x],
                False,
                True,
                original_type,
            )
    else:
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            return x, True, False, original_type
        else:
            return (
                x if isinstance(x, np.ndarray) else x.numpy(),
                False,
                False,
                original_type,
            )


def soft_threshold(
    x: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
    lambda_: float,
) -> Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]:
    """
    Soft thresholding operator (proximal operator for L1 norm).
    """
    x, is_torch, is_list, original_type = _process_input(x)

    if is_torch:

        def _soft_thresh(t):
            return torch.sign(t) * torch.clamp(torch.abs(t) - lambda_, min=0)
    else:

        def _soft_thresh(t):
            return np.sign(t) * np.maximum(np.abs(t) - lambda_, 0)

    if is_list:
        result = [_soft_thresh(t) for t in x]
    else:
        result = _soft_thresh(x)

    return result


def isotropic_soft_threshold(
    x: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
    lambda_: float,
) -> Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]:
    """
    Isotropic soft thresholding operator (proximal operator for L2 norm).
    prox_λ(x) = max(0, 1 - λ/||x||₂)x
    where ||x||₂ is computed over the entire input.
    """
    x, is_torch, is_list, original_type = _process_input(x)

    # Compute total norm once
    if is_torch:
        _fmax = lambda x: torch.clamp(x, min=0.0)

        if is_list:
            norm = torch.linalg.norm(torch.stack(x))
        else:
            norm = torch.norm(x)
    else:
        if is_list:
            norm = np.sqrt(sum(np.sum(t**2) for t in x))
        else:
            norm = np.linalg.norm(x)

    scale = 1 - lambda_ / norm

    # Apply scaling
    if is_list:
        result = [t * scale for t in x]
    else:
        result = x * scale

    return result


def scad_proximal(
    x: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
    sigma: float,
    tau: float = 1.0,
    a: float = 3.7,
) -> Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]:
    """
    SCAD proximal operator with correct regions and formulas:
    prox_τSCAD(x) = {
        sgn(x)max(0, |x| - σ),          |x| ≤ σ(a-1-τ+aτ)/(a-1)
        ((a-1)x - sgn(x)aτσ)/(a-1-τ),   σ(a-1-τ+aτ)/(a-1) < |x| ≤ aσ
        x,                               |x| > aσ
    }
    """
    x, is_torch, is_list, original_type = _process_input(x)

    # Compute threshold for first region
    threshold1 = sigma * (a - 1 - tau + a * tau) / (a - 1)
    threshold2 = a * sigma

    if is_torch:

        def _scad_prox(t):
            abs_t = torch.abs(t)
            sign_t = torch.sign(t)

            # Define regions
            region1 = abs_t <= threshold1
            region2 = (threshold1 < abs_t) & (abs_t <= threshold2)
            region3 = abs_t > threshold2

            result = torch.zeros_like(t)

            # Region 1: soft-thresholding
            result[region1] = sign_t[region1] * torch.clamp(
                abs_t[region1] - sigma, min=0
            )

            # Region 2: quadratic interpolation
            result[region2] = (
                (a - 1) * t[region2] - sign_t[region2] * a * tau * sigma
            ) / (a - 1 - tau)

            # Region 3: identity
            result[region3] = t[region3]

            return result
    else:

        def _scad_prox(t):
            abs_t = np.abs(t)
            sign_t = np.sign(t)

            # Define regions
            region1 = abs_t <= threshold1
            region2 = (threshold1 < abs_t) & (abs_t <= threshold2)
            region3 = abs_t > threshold2

            result = np.zeros_like(t)

            # Region 1: soft-thresholding
            result[region1] = sign_t[region1] * np.maximum(
                abs_t[region1] - sigma, 0
            )

            # Region 2: quadratic interpolation
            result[region2] = (
                (a - 1) * t[region2] - sign_t[region2] * a * tau * sigma
            ) / (a - 1 - tau)

            # Region 3: identity
            result[region3] = t[region3]

            return result

    if is_list:
        result = [_scad_prox(t) for t in x]
    else:
        result = _scad_prox(x)

    return result


def isotropic_scad_proximal(
    x: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
    sigma: float,
    tau: float = 1.0,
    a: float = 3.7,
) -> Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]:
    """
    Isotropic SCAD proximal operator with correct thresholds and formulas.
    Applies the SCAD proximal operator to the norm of the entire input:
    prox_τSCAD(x) = {
        max(0, 1 - σ/||x||₂)x,                    ||x||₂ ≤ σ(a-1-τ+aτ)/(a-1)
        ((a-1)x - (aτσx/||x||₂))/(a-1-τ),        σ(a-1-τ+aτ)/(a-1) < ||x||₂ ≤ aσ
        x,                                        ||x||₂ > aσ
    }
    Where ||x||₂ is computed over the entire input (including all tensors if input is a list).
    """
    x, is_torch, is_list, original_type = _process_input(x)

    # Compute thresholds
    threshold1 = sigma * (a - 1 - tau + a * tau) / (a - 1)
    threshold2 = a * sigma

    # Compute total norm once
    if is_torch:
        if is_list:
            norm = torch.sqrt(sum(torch.sum(t**2) for t in x))
        else:
            norm = torch.norm(x)

        # Initialize scaling factor
        if norm <= threshold1:
            # Region 1: soft-thresholding
            scale = torch.clamp(1 - sigma / norm, min=0)
        elif norm <= threshold2:
            # Region 2: quadratic interpolation
            scale = ((a - 1) - a * tau * sigma / norm) / (a - 1 - tau)
        else:
            # Region 3: identity
            scale = 1.0
    else:
        if is_list:
            norm = np.sqrt(sum(np.sum(t**2) for t in x))
        else:
            norm = np.linalg.norm(x)

        # Initialize scaling factor
        if norm <= threshold1:
            # Region 1: soft-thresholding
            scale = max(1 - sigma / norm, 0)
        elif norm <= threshold2:
            # Region 2: quadratic interpolation
            scale = ((a - 1) - a * tau * sigma / norm) / (a - 1 - tau)
        else:
            # Region 3: identity
            scale = 1.0

    # Apply scaling
    if is_list:
        result = [t * scale for t in x]
    else:
        result = x * scale

    return result
