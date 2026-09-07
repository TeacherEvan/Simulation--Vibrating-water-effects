"""SPH smoothing kernels (Muller et al. 2003) - pure NumPy reference.

Poly6 - density kernel (C^2 continuous, non-negative, used for density + pressure).
Spiky - pressure-gradient kernel (gradient points along r, used for forces).
Viscosity - Laplacian kernel (used for viscous diffusion).

All kernels take pairwise distances r (ndarray) and smoothing length h.
"""

from __future__ import annotations

import numpy as np

# 315 / (64 * pi * h^9)
_POLY6_C0 = 315.0 / (64.0 * np.pi)
# -45 / (pi * h^6)
_SPIKY_GRAD_C0 = -45.0 / np.pi
# 45 / (pi * h^6)
_VISC_LAP_C0 = 45.0 / np.pi


def poly6(r: np.ndarray, h: float) -> np.ndarray:
    """Poly6 smoothing kernel W(r, h).

    Returns zero where r >= h (no support outside).
    """
    h9 = h ** 9
    diff = (h * h - r * r)
    diff = np.maximum(diff, 0.0)
    return _POLY6_C0 / h9 * diff ** 3


def poly6_constant(h: float) -> float:
    """Poly6 evaluated at r = 0 (peak density kernel weight = 315/(64*pi*h^3))."""
    return _POLY6_C0 / (h ** 3)


def spiky_gradient(r: np.ndarray, h: float) -> np.ndarray:
    """Magnitude of the spiky kernel gradient (used along r_hat).

    Returns 0 where r == 0 to avoid singularity (caller is responsible
    for applying the r_hat direction).
    """
    out = np.zeros_like(r)
    mask = (r > 0.0) & (r < h)
    diff = h - r[mask]
    out[mask] = _SPIKY_GRAD_C0 / (h ** 6) * diff ** 2
    return out


def viscosity_laplacian(r: np.ndarray, h: float) -> np.ndarray:
    """Viscosity kernel Laplacian (scalar; returned for each pair)."""
    out = np.zeros_like(r)
    mask = r < h
    out[mask] = _VISC_LAP_C0 / (h ** 6) * (h - r[mask])
    return out
