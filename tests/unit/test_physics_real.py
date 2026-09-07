"""Real (non-mock) tests for src.physics.NumpySPHSolver and TaichiSPHSolver.

These exercise the actual implementation. They are intentionally small
so they fit inside the 30-min wall budget even on the O(N^2) path.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.physics import (  # noqa: E402
    NumpySPHSolver,
    SPHConfig,
    TaichiSPHSolver,
    poly6,
    poly6_constant,
    spiky_gradient,
    viscosity_laplacian,
)


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------
class TestKernels:
    def test_poly6_at_zero_is_peak(self):
        h = 0.05
        val = poly6(np.array([0.0]), h)[0]
        assert math.isclose(val, poly6_constant(h), rel_tol=1e-9)

    def test_poly6_zero_beyond_support(self):
        h = 0.05
        assert poly6(np.array([h + 1e-6]), h)[0] == 0.0

    def test_spiky_gradient_zero_at_zero(self):
        h = 0.05
        assert spiky_gradient(np.array([0.0]), h)[0] == 0.0

    def test_viscosity_laplacian_zero_beyond_support(self):
        h = 0.05
        assert viscosity_laplacian(np.array([h + 1e-6]), h)[0] == 0.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
class TestSPHConfig:
    def test_defaults_are_valid(self):
        c = SPHConfig()
        assert c.particle_count > 0
        assert c.container_radius > 0

    def test_invalid_particle_count_rejected(self):
        with pytest.raises(ValueError):
            SPHConfig(particle_count=0)


# ---------------------------------------------------------------------------
# NumpySPHSolver tests
# ---------------------------------------------------------------------------
class TestNumpySPHSolver:
    def _solver(self, n: int = 100) -> NumpySPHSolver:
        return NumpySPHSolver(SPHConfig(particle_count=n))

    def test_init_shapes(self):
        s = self._solver(50)
        assert s.positions.shape == (50, 3)
        assert s.velocities.shape == (50, 3)
        assert s.densities.shape == (50,)
        assert s.pressures.shape == (50,)

    def test_step_runs(self):
        s = self._solver(50)
        for _ in range(10):
            s.step()
        assert s.step_count == 10

    def test_particles_remain_inside_container(self):
        s = self._solver(100)
        for _ in range(100):
            s.step()
        norms = np.linalg.norm(s.positions, axis=1)
        assert norms.max() <= s.config.container_radius + 1e-3, (
            f"escaped: max_r={norms.max()} R={s.config.container_radius}"
        )

    def test_speed_cap_holds(self):
        s = self._solver(100)
        for _ in range(100):
            s.step()
        speed = np.linalg.norm(s.velocities, axis=1)
        assert speed.max() <= s.config.speed_of_light_max + 1e-9

    def test_1000_steps_no_crash(self):
        s = self._solver(100)
        for _ in range(1000):
            s.step()
        assert s.step_count == 1000


# ---------------------------------------------------------------------------
# TaichiSPHSolver tests (CPU fallback path is the main verification)
# ---------------------------------------------------------------------------
class TestTaichiSPHSolver:
    def test_init_runs(self):
        ts = TaichiSPHSolver(SPHConfig(particle_count=20))
        assert ts.init_arch in {"cpu", "cpu-fallback", "no-taichi"}

    def test_step_via_fallback(self):
        ts = TaichiSPHSolver(SPHConfig(particle_count=20))
        ts.step()
        ts.step()
        assert ts.positions.shape == (20, 3)
