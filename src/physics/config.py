"""Configuration dataclass for SPH solvers.

Author: surgical-implementation run 2026-09-07.
All fields use SI-ish units suitable for a 1m-diameter water-filled
spherical glass container. Defaults are tuned for stability, not realism.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SPHConfig:
    """Static configuration consumed by NumpySPHSolver / TaichiSPHSolver.

    Attributes
    ----------
    particle_count : int
        Number of fluid particles. Defaults to 1000 (CPU-friendly).
    container_radius : float
        Inner radius of the spherical container in meters.
    smoothing_length : float
        SPH support radius h (m). Default 0.05 (~5cm).
    rest_density : float
        Target density (kg/m^3). Water ~= 1000.
    stiffness : float
        Equation-of-state stiffness for Tait-style pressure.
    viscosity : float
        Dynamic viscosity coefficient.
    gravity : tuple[float, float, float]
        Constant acceleration applied to every particle (m/s^2).
    speed_of_light_max : float
        Hard cap on |v| (m/s). Stops runaway integration.
    dt : float
        Default timestep in seconds; may be overridden per-step().
    """

    particle_count: int = 1000
    container_radius: float = 0.5
    smoothing_length: float = 0.05
    rest_density: float = 1000.0
    stiffness: float = 500.0
    viscosity: float = 0.05
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    speed_of_light_max: float = 50.0
    dt: float = 1.0 / 120.0

    def __post_init__(self) -> None:
        if self.particle_count <= 0:
            raise ValueError("particle_count must be > 0")
        if self.container_radius <= 0.0:
            raise ValueError("container_radius must be > 0")
        if self.smoothing_length <= 0.0:
            raise ValueError("smoothing_length must be > 0")
        if self.speed_of_light_max <= 0.0:
            raise ValueError("speed_of_light_max must be > 0")
