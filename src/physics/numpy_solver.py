"""CPU reference SPH solver.

A small, dependency-light NumpySPHSolver sufficient for unit tests and
headless validation. Naive O(N^2) neighbour search - intended for <=~5k
particles.

Stability choices for this minimal reference:
- Confining inward force when r > 0.9 * R (soft wall) plus a hard clamp
  to R - eps after every step.
- Speed cap to c_max.
- Pressure clamped to non-negative (Tait-style).
- Default timestep small (1/120 s) so the O(N^2) inner loop stays
  tractable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import SPHConfig
from .kernels import poly6, spiky_gradient, viscosity_laplacian


class NumpySPHSolver:
    """Reference CPU SPH solver.

    Public attributes (numpy arrays, shape (N, 3) unless noted):
        positions
        velocities
        densities  (shape (N,))
        pressures  (shape (N,))
    """

    def __init__(self, config: Optional[SPHConfig] = None) -> None:
        self.config = config or SPHConfig()
        self.positions = np.zeros((self.config.particle_count, 3), dtype=np.float64)
        self.velocities = np.zeros_like(self.positions)
        self.densities = np.zeros(self.config.particle_count, dtype=np.float64)
        self.pressures = np.zeros(self.config.particle_count, dtype=np.float64)
        self._step_count = 0
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """(Re-)initialise particles inside the spherical container."""
        rng = np.random.default_rng(seed=42)
        n = self.config.particle_count
        R = self.config.container_radius * 0.5
        pts: list[np.ndarray] = []
        needed = n
        while needed > 0:
            cand = rng.uniform(-R, R, size=(needed * 2, 3))
            norms = np.linalg.norm(cand, axis=1)
            inside = cand[norms <= R]
            pts.append(inside[:needed])
            needed -= len(inside)
        self.positions = np.concatenate(pts)[:n].astype(np.float64)
        self.velocities = rng.normal(0.0, 0.01, size=(n, 3)).astype(np.float64)
        self.densities = np.zeros(n, dtype=np.float64)
        self.pressures = np.zeros(n, dtype=np.float64)
        self._step_count = 0

    def step(self, dt: Optional[float] = None) -> None:
        """Advance one timestep using semi-implicit Euler with SPH forces."""
        if dt is None:
            dt = self.config.dt
        self._compute_density_pressure()
        accelerations = self._compute_accelerations()
        # semi-implicit Euler
        self.velocities = self.velocities + accelerations * dt
        self._apply_speed_cap()
        self.positions = self.positions + self.velocities * dt
        self._enforce_boundary()
        self._step_count += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_density_pressure(self) -> None:
        h = self.config.smoothing_length
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        w = poly6(r, h)
        vol = (4.0 / 3.0) * np.pi * (self.config.container_radius ** 3)
        m_i = self.config.rest_density * vol / self.config.particle_count
        self.densities = m_i * w.sum(axis=1)
        self.pressures = self.config.stiffness * np.maximum(
            self.densities - self.config.rest_density, 0.0
        )

    def _compute_accelerations(self) -> np.ndarray:
        cfg = self.config
        h = cfg.smoothing_length
        n = cfg.particle_count
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        r_safe = np.where(r > 0, r, 1.0)
        spiky = spiky_gradient(r, h)
        visc = viscosity_laplacian(r, h)
        vol = (4.0 / 3.0) * np.pi * (cfg.container_radius ** 3)
        m_j = cfg.rest_density * vol / n
        rho_i = self.densities[:, None] + 1e-12
        rho_j = self.densities[None, :] + 1e-12
        term = (
            self.pressures[:, None] / (rho_i ** 2)
            + self.pressures[None, :] / (rho_j ** 2)
        )
        diff_hat = diff / r_safe[..., None]
        a_press = -np.sum(
            m_j * term[..., None] * spiky[..., None] * diff_hat, axis=1
        )
        dv = self.velocities[None, :, :] - self.velocities[:, None, :]
        a_visc = cfg.viscosity * np.sum(
            m_j * dv * visc[..., None], axis=1
        ) / (rho_i ** 2)
        # Soft confining inward force when |x| > 0.85 * R (avoids escape).
        norms = np.linalg.norm(self.positions, axis=1)
        a_conf = np.zeros_like(self.positions)
        wall = 0.85 * cfg.container_radius
        outside = norms > wall
        if np.any(outside):
            n_hat = self.positions[outside] / norms[outside, None]
            # Strength grows linearly with how far past wall we are.
            strength = 50.0 * (norms[outside] - wall) / (cfg.container_radius - wall)
            a_conf[outside] = -cfg.container_radius * n_hat * strength[:, None]
        g = np.array(cfg.gravity, dtype=np.float64)
        return a_press + a_visc + a_conf + g

    def _apply_speed_cap(self) -> None:
        c = self.config.speed_of_light_max
        speed = np.linalg.norm(self.velocities, axis=1)
        too_fast = speed > c
        if np.any(too_fast):
            scale = (c / speed[too_fast])[..., None]
            self.velocities[too_fast] = self.velocities[too_fast] * scale

    def _enforce_boundary(self) -> None:
        R = self.config.container_radius
        eps = 1e-3
        norms = np.linalg.norm(self.positions, axis=1)
        outside = norms > R - eps
        if np.any(outside):
            n_hat = self.positions[outside] / norms[outside, None]
            self.positions[outside] = n_hat * (R - eps)
            v_n = np.sum(self.velocities[outside] * n_hat, axis=1, keepdims=True)
            # remove inward velocity component (restitution ~ 0)
            self.velocities[outside] = self.velocities[outside] - v_n * n_hat

    # convenience for tests
    @property
    def step_count(self) -> int:
        return self._step_count
