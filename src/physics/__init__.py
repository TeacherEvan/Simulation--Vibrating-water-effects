"""src.physics - SPH fluid simulation package.

Public symbols:
    SPHConfig
    NumpySPHSolver
    TaichiSPHSolver
    kernels (poly6, poly6_constant, spiky_gradient, viscosity_laplacian)
"""

from .config import SPHConfig
from .kernels import (
    poly6,
    poly6_constant,
    spiky_gradient,
    viscosity_laplacian,
)
from .numpy_solver import NumpySPHSolver
from .taichi_solver import TaichiSPHSolver

__all__ = [
    "SPHConfig",
    "NumpySPHSolver",
    "TaichiSPHSolver",
    "poly6",
    "poly6_constant",
    "spiky_gradient",
    "viscosity_laplacian",
]
