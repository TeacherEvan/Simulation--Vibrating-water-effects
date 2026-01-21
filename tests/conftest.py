"""
Pytest Configuration and Shared Fixtures
==========================================

This module provides shared test fixtures, configuration, and utilities
used across all test categories (unit, integration, system, acceptance).

Fixture Scopes:
- function: Created fresh for each test function (default)
- class: Created once per test class
- module: Created once per test module
- session: Created once per test session

Usage:
    @pytest.fixture
    def my_fixture():
        return something
    
    def test_something(my_fixture):
        assert my_fixture.works()
"""

import pytest
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@dataclass
class SimulationConfig:
    """Test configuration for simulation parameters."""
    
    particle_count: int = 1000
    tank_radius: float = 1.0
    time_step: float = 0.001
    gravity: tuple = (0.0, -9.81, 0.0)
    viscosity: float = 0.01
    rest_density: float = 1000.0
    speed_of_light_limit: float = 299792458.0 * 0.999
    use_gpu: bool = False


@pytest.fixture
def default_config() -> SimulationConfig:
    """
    Provide default simulation configuration for testing.
    
    Returns:
        SimulationConfig with safe default values for testing.
    
    Example:
        def test_simulation(default_config):
            sim = Simulation(default_config)
            assert sim.particle_count == 1000
    """
    return SimulationConfig()


@pytest.fixture
def gpu_config() -> SimulationConfig:
    """Configuration for GPU-accelerated tests (10k particles)."""
    return SimulationConfig(
        particle_count=10000,
        use_gpu=True
    )


@pytest.fixture
def minimal_config() -> SimulationConfig:
    """Minimal configuration for fast unit tests."""
    return SimulationConfig(
        particle_count=100,
        time_step=0.01
    )


@pytest.fixture
def high_precision_config() -> SimulationConfig:
    """High precision configuration for validation tests."""
    return SimulationConfig(
        particle_count=500,
        time_step=0.0001
    )


# =============================================================================
# Mock Objects and Stubs
# =============================================================================

@pytest.fixture
def mock_panda3d_base():
    """
    Mock Panda3D ShowBase for testing without graphics.
    
    Returns:
        Mock object simulating Panda3D ShowBase interface.
    
    Expected Behavior:
        - win.getXSize() returns 1920
        - win.getYSize() returns 1080
        - cam.getPos() returns (0, -10, 5)
    """
    mock_base = MagicMock()
    mock_base.win.getXSize.return_value = 1920
    mock_base.win.getYSize.return_value = 1080
    mock_base.cam.getPos.return_value = MagicMock(x=0, y=-10, z=5)
    mock_base.render.attach_new_node.return_value = MagicMock()
    return mock_base


@pytest.fixture
def mock_taichi():
    """
    Mock Taichi library for testing without GPU.
    
    Returns:
        Mock object simulating Taichi interface.
    
    Note:
        Use this when testing logic that uses Taichi without
        needing actual GPU execution.
    """
    with patch.dict(sys.modules, {'taichi': MagicMock()}):
        yield sys.modules['taichi']


@pytest.fixture
def mock_hdf5_file(tmp_path):
    """
    Create a mock HDF5 file for data logging tests.
    
    Returns:
        Path to temporary HDF5 file.
    """
    import h5py
    file_path = tmp_path / "test_data.h5"
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('positions', shape=(100, 3), dtype='float32')
        f.create_dataset('velocities', shape=(100, 3), dtype='float32')
        f.attrs['particle_count'] = 100
    return file_path


@pytest.fixture
def mock_sqlite_db(tmp_path):
    """
    Create a mock SQLite database for event logging tests.
    
    Returns:
        Path to temporary SQLite database.
    """
    import sqlite3
    db_path = tmp_path / "test_events.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            timestamp REAL,
            event_type TEXT,
            data TEXT
        )
    ''')
    conn.commit()
    conn.close()
    return db_path


# =============================================================================
# Physics Test Data
# =============================================================================

@pytest.fixture
def sample_particles() -> Dict[str, np.ndarray]:
    """
    Generate sample particle data for physics tests.
    
    Returns:
        Dictionary containing positions, velocities, and densities.
    
    Example:
        def test_physics(sample_particles):
            positions = sample_particles['positions']
            assert positions.shape == (100, 3)
    """
    np.random.seed(42)  # Reproducible tests
    n_particles = 100
    return {
        'positions': np.random.uniform(-1, 1, (n_particles, 3)).astype(np.float32),
        'velocities': np.random.uniform(-0.1, 0.1, (n_particles, 3)).astype(np.float32),
        'densities': np.ones(n_particles, dtype=np.float32) * 1000.0,
        'pressures': np.zeros(n_particles, dtype=np.float32),
        'masses': np.ones(n_particles, dtype=np.float32) * 0.01
    }


@pytest.fixture
def spherical_tank_particles() -> Dict[str, np.ndarray]:
    """
    Generate particles arranged in a spherical tank.
    
    Returns:
        Particles positioned within a unit sphere.
    """
    np.random.seed(42)
    n_particles = 500
    
    # Generate points uniformly in a sphere using rejection sampling
    points = []
    while len(points) < n_particles:
        candidate = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(candidate) <= 0.95:  # Inside tank with margin
            points.append(candidate)
    
    positions = np.array(points, dtype=np.float32)
    
    return {
        'positions': positions,
        'velocities': np.zeros_like(positions),
        'densities': np.ones(n_particles, dtype=np.float32) * 1000.0
    }


@pytest.fixture
def boundary_particles() -> Dict[str, np.ndarray]:
    """
    Generate particles at boundary conditions for edge case testing.
    
    Returns:
        Particles positioned at tank boundaries.
    """
    # Particles at exact boundary, just inside, and just outside
    positions = np.array([
        [0.0, 0.0, 0.0],      # Center
        [1.0, 0.0, 0.0],      # Exact boundary
        [0.999, 0.0, 0.0],    # Just inside
        [1.001, 0.0, 0.0],    # Just outside
        [-1.0, 0.0, 0.0],     # Opposite boundary
    ], dtype=np.float32)
    
    return {
        'positions': positions,
        'velocities': np.zeros_like(positions),
        'densities': np.ones(len(positions), dtype=np.float32) * 1000.0
    }


# =============================================================================
# Visual Effects Test Data
# =============================================================================

@pytest.fixture
def vfx_config() -> Dict[str, Any]:
    """
    Configuration for visual effects testing.
    
    Returns:
        Dictionary with VFX settings.
    """
    return {
        'bloom_enabled': True,
        'bloom_threshold': 1.0,
        'bloom_intensity': 0.8,
        'bloom_iterations': 3,
        'ssao_enabled': True,
        'ssao_samples': 16,
        'ssao_radius': 0.5,
        'ssao_intensity': 1.0,
        'ssr_enabled': False,
        'dof_enabled': False,
        'tone_mapping': 'aces',
        'exposure': 1.0,
        'gamma': 2.2,
        'target_fps': 60,
        'dynamic_quality': False
    }


@pytest.fixture
def mock_textures():
    """
    Mock texture objects for shader tests.
    
    Returns:
        Dictionary of mock texture objects.
    """
    return {
        'scene': MagicMock(name='scene_texture'),
        'depth': MagicMock(name='depth_texture'),
        'normal': MagicMock(name='normal_texture'),
        'bloom': MagicMock(name='bloom_texture'),
        'ssao': MagicMock(name='ssao_texture')
    }


# =============================================================================
# Temporary Directory and File Fixtures
# =============================================================================

@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """
    Create temporary directory for test data files.
    
    Returns:
        Path to temporary data directory.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_shader_dir(tmp_path) -> Path:
    """
    Create temporary directory with mock shader files.
    
    Returns:
        Path to temporary shader directory.
    """
    shader_dir = tmp_path / "shaders"
    shader_dir.mkdir()
    
    # Create mock shader files
    (shader_dir / "water").mkdir()
    (shader_dir / "glass").mkdir()
    (shader_dir / "post_process").mkdir()
    (shader_dir / "particles").mkdir()
    
    # Create minimal mock shader content
    mock_vert = "void main() { gl_Position = vec4(0.0); }"
    mock_frag = "void main() { gl_FragColor = vec4(1.0); }"
    
    (shader_dir / "post_process" / "fullscreen_quad.vert").write_text(mock_vert)
    (shader_dir / "post_process" / "bloom_bright.frag").write_text(mock_frag)
    (shader_dir / "post_process" / "gaussian_blur.frag").write_text(mock_frag)
    (shader_dir / "post_process" / "ssao.frag").write_text(mock_frag)
    (shader_dir / "post_process" / "tone_mapping.frag").write_text(mock_frag)
    
    return shader_dir


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_timer():
    """
    Timer fixture for performance measurements.
    
    Returns:
        Timer context manager for benchmarking.
    """
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time
    
    return Timer


# =============================================================================
# Validation and Assertion Helpers
# =============================================================================

@pytest.fixture
def physics_validator():
    """
    Physics validation helper for checking conservation laws.
    
    Returns:
        Validator object with physics-specific assertions.
    """
    class PhysicsValidator:
        @staticmethod
        def check_energy_conservation(
            initial_energy: float,
            final_energy: float,
            tolerance: float = 0.01
        ) -> bool:
            """Check that energy is conserved within tolerance."""
            relative_error = abs(final_energy - initial_energy) / max(initial_energy, 1e-10)
            return relative_error <= tolerance
        
        @staticmethod
        def check_momentum_conservation(
            initial_momentum: np.ndarray,
            final_momentum: np.ndarray,
            tolerance: float = 1e-6
        ) -> bool:
            """Check that momentum is conserved within tolerance."""
            diff = np.linalg.norm(final_momentum - initial_momentum)
            return diff <= tolerance
        
        @staticmethod
        def check_particles_in_bounds(
            positions: np.ndarray,
            bounds: float = 1.0
        ) -> bool:
            """Check all particles are within spherical bounds."""
            distances = np.linalg.norm(positions, axis=1)
            return np.all(distances <= bounds)
        
        @staticmethod
        def check_velocity_limit(
            velocities: np.ndarray,
            max_speed: float
        ) -> bool:
            """Check velocities don't exceed speed limit."""
            speeds = np.linalg.norm(velocities, axis=1)
            return np.all(speeds <= max_speed)
    
    return PhysicsValidator()


@pytest.fixture
def numerical_validator():
    """
    Numerical validation helper for checking simulation stability.
    
    Returns:
        Validator with numerical stability checks.
    """
    class NumericalValidator:
        @staticmethod
        def check_no_nan(array: np.ndarray) -> bool:
            """Check array contains no NaN values."""
            return not np.any(np.isnan(array))
        
        @staticmethod
        def check_no_inf(array: np.ndarray) -> bool:
            """Check array contains no Inf values."""
            return not np.any(np.isinf(array))
        
        @staticmethod
        def check_finite(array: np.ndarray) -> bool:
            """Check all values are finite."""
            return np.all(np.isfinite(array))
        
        @staticmethod
        def check_positive(array: np.ndarray) -> bool:
            """Check all values are positive."""
            return np.all(array > 0)
    
    return NumericalValidator()


# =============================================================================
# Cleanup and Session-Scoped Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def session_temp_dir(tmp_path_factory) -> Path:
    """
    Session-scoped temporary directory for shared test files.
    
    Returns:
        Path to session temporary directory.
    """
    return tmp_path_factory.mktemp("session_data")


@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Reset random seeds before each test for reproducibility.
    
    Note:
        This fixture runs automatically before every test.
    """
    np.random.seed(42)


# =============================================================================
# Skip Markers and Conditions
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    # Markers are defined in pytest.ini, this is for dynamic configuration
    pass


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add skip markers based on environment.
    """
    # Check for GPU availability
    gpu_available = False
    try:
        import taichi as ti
        ti.init(arch=ti.gpu, offline_cache=False)
        gpu_available = True
    except (ImportError, RuntimeError):
        pass
    
    # Check for Panda3D availability
    panda3d_available = False
    try:
        import panda3d.core
        panda3d_available = True
    except ImportError:
        pass
    
    skip_gpu = pytest.mark.skip(reason="GPU/Taichi not available")
    skip_panda3d = pytest.mark.skip(reason="Panda3D not installed")
    
    for item in items:
        if "gpu" in item.keywords and not gpu_available:
            item.add_marker(skip_gpu)
        if "panda3d" in item.keywords and not panda3d_available:
            item.add_marker(skip_panda3d)
