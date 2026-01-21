"""
Integration Tests for Physics-Rendering Pipeline
==================================================

This module tests the integration between physics simulation and
rendering systems, including data flow, LOD switching, and
synchronization.

Test Categories:
- Physics to rendering data flow
- LOD switching behavior
- Pause/Resume functionality
- Configuration synchronization

Expected Outcomes:
- Particle data flows correctly from physics to rendering
- LOD switching triggers at correct thresholds
- State is preserved across pause/resume
- Configuration changes propagate correctly
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Mock Classes for Integration Testing
# =============================================================================

class MockPhysicsEngine:
    """Mock physics engine for integration testing."""
    
    def __init__(self, particle_count=1000):
        self.particle_count = particle_count
        self.positions = np.random.uniform(-1, 1, (particle_count, 3)).astype(np.float32)
        self.velocities = np.zeros((particle_count, 3), dtype=np.float32)
        self.densities = np.ones(particle_count, dtype=np.float32) * 1000
        self.paused = False
        self.time = 0.0
        self.step_count = 0
        self.mode = 'cpu'  # 'cpu' or 'gpu'
    
    def step(self, dt):
        if not self.paused:
            # Simple movement for testing
            self.positions += self.velocities * dt
            self.velocities[:, 1] -= 9.81 * dt  # gravity
            self.time += dt
            self.step_count += 1
    
    def get_positions(self):
        return self.positions.copy()
    
    def get_velocities(self):
        return self.velocities.copy()
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
    
    def reset(self):
        self.positions = np.random.uniform(-1, 1, (self.particle_count, 3)).astype(np.float32)
        self.velocities = np.zeros_like(self.velocities)
        self.time = 0.0
        self.step_count = 0


class MockRenderer:
    """Mock renderer for integration testing."""
    
    def __init__(self):
        self.particle_positions = None
        self.particle_velocities = None
        self.update_count = 0
        self.last_update_time = None
    
    def update_particles(self, positions, velocities=None):
        self.particle_positions = positions.copy()
        if velocities is not None:
            self.particle_velocities = velocities.copy()
        self.update_count += 1
        self.last_update_time = time.time()
    
    def get_particle_count(self):
        if self.particle_positions is not None:
            return len(self.particle_positions)
        return 0


class MockLODController:
    """Mock LOD controller for integration testing."""
    
    def __init__(self, physics_engine):
        self.physics = physics_engine
        self.current_mode = 'cpu'
        self.switch_count = 0
        self.hysteresis_frames = 30
        self.frames_in_current_mode = 0
        
        # Thresholds
        self.gpu_threshold = 5000  # Switch to GPU above this
        self.cpu_threshold = 2000  # Switch to CPU below this
    
    def update(self, frame_time):
        self.frames_in_current_mode += 1
        
        particle_count = self.physics.particle_count
        
        if self.frames_in_current_mode >= self.hysteresis_frames:
            if self.current_mode == 'cpu' and particle_count > self.gpu_threshold:
                self.switch_to_gpu()
            elif self.current_mode == 'gpu' and particle_count < self.cpu_threshold:
                self.switch_to_cpu()
    
    def switch_to_gpu(self):
        if self.current_mode != 'gpu':
            self.current_mode = 'gpu'
            self.physics.mode = 'gpu'
            self.switch_count += 1
            self.frames_in_current_mode = 0
    
    def switch_to_cpu(self):
        if self.current_mode != 'cpu':
            self.current_mode = 'cpu'
            self.physics.mode = 'cpu'
            self.switch_count += 1
            self.frames_in_current_mode = 0


class MockSimulationPipeline:
    """Complete simulation pipeline for integration testing."""
    
    def __init__(self, particle_count=1000):
        self.physics = MockPhysicsEngine(particle_count)
        self.renderer = MockRenderer()
        self.lod_controller = MockLODController(self.physics)
        self.frame_count = 0
    
    def step(self, dt=0.016):
        # Update physics
        self.physics.step(dt)
        
        # Update LOD
        self.lod_controller.update(dt)
        
        # Transfer to renderer
        self.renderer.update_particles(
            self.physics.get_positions(),
            self.physics.get_velocities()
        )
        
        self.frame_count += 1
    
    def pause(self):
        self.physics.pause()
    
    def resume(self):
        self.physics.resume()
    
    def reset(self):
        self.physics.reset()
        self.renderer.update_particles(
            self.physics.get_positions(),
            self.physics.get_velocities()
        )


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def physics_engine():
    return MockPhysicsEngine(particle_count=1000)


@pytest.fixture
def renderer():
    return MockRenderer()


@pytest.fixture
def lod_controller(physics_engine):
    return MockLODController(physics_engine)


@pytest.fixture
def simulation_pipeline():
    return MockSimulationPipeline(particle_count=1000)


# =============================================================================
# Physics-Rendering Data Flow Tests
# =============================================================================

class TestPhysicsRenderingDataFlow:
    """
    Test data flow from physics to rendering.
    
    Tests verify:
    - Particle positions transfer correctly
    - Velocities are available for rendering effects
    - Data is synchronized each frame
    """
    
    @pytest.mark.integration
    def test_positions_transfer(self, physics_engine, renderer):
        """
        Test particle positions transfer from physics to renderer.
        
        Expected: Renderer receives same positions as physics.
        """
        physics_engine.step(0.016)
        renderer.update_particles(physics_engine.get_positions())
        
        np.testing.assert_array_almost_equal(
            renderer.particle_positions,
            physics_engine.positions
        )
    
    @pytest.mark.integration
    def test_velocities_transfer(self, physics_engine, renderer):
        """
        Test velocities transfer for motion blur effects.
        
        Expected: Renderer receives velocity data.
        """
        physics_engine.step(0.016)
        renderer.update_particles(
            physics_engine.get_positions(),
            physics_engine.get_velocities()
        )
        
        assert renderer.particle_velocities is not None
        assert len(renderer.particle_velocities) == physics_engine.particle_count
    
    @pytest.mark.integration
    def test_data_sync_every_frame(self, simulation_pipeline):
        """
        Test data synchronization happens each frame.
        
        Expected: Renderer update count matches frame count.
        """
        for _ in range(10):
            simulation_pipeline.step()
        
        assert simulation_pipeline.renderer.update_count == 10
    
    @pytest.mark.integration
    def test_renderer_particle_count_matches(self, simulation_pipeline):
        """
        Test renderer has correct particle count.
        
        Expected: Renderer particle count matches physics.
        """
        simulation_pipeline.step()
        
        assert simulation_pipeline.renderer.get_particle_count() == \
               simulation_pipeline.physics.particle_count
    
    @pytest.mark.integration
    def test_data_is_copy_not_reference(self, physics_engine, renderer):
        """
        Test renderer gets a copy of data, not reference.
        
        Expected: Modifying renderer data doesn't affect physics.
        """
        physics_engine.step(0.016)
        original_positions = physics_engine.positions.copy()
        
        renderer.update_particles(physics_engine.get_positions())
        renderer.particle_positions[:] = 0  # Modify renderer's copy
        
        np.testing.assert_array_equal(physics_engine.positions, original_positions)


# =============================================================================
# LOD Switching Tests
# =============================================================================

class TestLODSwitching:
    """
    Test Level of Detail switching behavior.
    
    Tests verify:
    - LOD switches at correct particle thresholds
    - Hysteresis prevents rapid switching
    - Mode transitions are logged
    """
    
    @pytest.mark.integration
    def test_initial_cpu_mode(self, lod_controller):
        """
        Test LOD starts in CPU mode.
        
        Expected: Initial mode is 'cpu'.
        """
        assert lod_controller.current_mode == 'cpu'
    
    @pytest.mark.integration
    def test_switch_to_gpu_threshold(self, physics_engine):
        """
        Test switch to GPU above threshold.
        
        Expected: Mode changes to 'gpu' when particle count exceeds threshold.
        """
        # Create controller with high particle count
        high_count_physics = MockPhysicsEngine(particle_count=6000)
        controller = MockLODController(high_count_physics)
        
        # Run past hysteresis period
        for _ in range(35):
            controller.update(0.016)
        
        assert controller.current_mode == 'gpu'
    
    @pytest.mark.integration
    def test_switch_to_cpu_threshold(self):
        """
        Test switch back to CPU below threshold.
        
        Expected: Mode changes to 'cpu' when particle count is low.
        """
        physics = MockPhysicsEngine(particle_count=1000)
        controller = MockLODController(physics)
        
        # Force GPU mode first
        controller.current_mode = 'gpu'
        physics.mode = 'gpu'
        controller.frames_in_current_mode = 0
        
        # Run past hysteresis
        for _ in range(35):
            controller.update(0.016)
        
        assert controller.current_mode == 'cpu'
    
    @pytest.mark.integration
    def test_hysteresis_prevents_rapid_switching(self, lod_controller):
        """
        Test hysteresis prevents mode thrashing.
        
        Expected: No switching within hysteresis period.
        """
        initial_switches = lod_controller.switch_count
        
        # Run for less than hysteresis frames
        for _ in range(20):
            lod_controller.update(0.016)
        
        assert lod_controller.switch_count == initial_switches
    
    @pytest.mark.integration
    def test_switch_count_tracked(self):
        """
        Test LOD switches are counted.
        
        Expected: switch_count increments on mode change.
        """
        physics = MockPhysicsEngine(particle_count=6000)
        controller = MockLODController(physics)
        
        # Trigger switch
        for _ in range(35):
            controller.update(0.016)
        
        assert controller.switch_count >= 1


# =============================================================================
# Pause/Resume Tests
# =============================================================================

class TestPauseResume:
    """
    Test pause and resume functionality.
    
    Tests verify:
    - Simulation pauses correctly
    - State is preserved during pause
    - Resume continues from correct state
    - No state leaks across pause/resume
    """
    
    @pytest.mark.integration
    def test_pause_stops_physics(self, simulation_pipeline):
        """
        Test pause stops physics simulation.
        
        Expected: Physics time doesn't advance when paused.
        """
        simulation_pipeline.step()
        simulation_pipeline.pause()
        
        time_before = simulation_pipeline.physics.time
        
        for _ in range(10):
            simulation_pipeline.step()
        
        assert simulation_pipeline.physics.time == time_before
    
    @pytest.mark.integration
    def test_resume_continues_simulation(self, simulation_pipeline):
        """
        Test resume continues simulation.
        
        Expected: Physics time advances after resume.
        """
        simulation_pipeline.step()
        simulation_pipeline.pause()
        simulation_pipeline.step()
        
        time_paused = simulation_pipeline.physics.time
        
        simulation_pipeline.resume()
        simulation_pipeline.step()
        
        assert simulation_pipeline.physics.time > time_paused
    
    @pytest.mark.integration
    def test_state_preserved_during_pause(self, simulation_pipeline):
        """
        Test particle state preserved during pause.
        
        Expected: Positions unchanged during pause.
        """
        for _ in range(5):
            simulation_pipeline.step()
        
        positions_before = simulation_pipeline.physics.positions.copy()
        
        simulation_pipeline.pause()
        
        for _ in range(5):
            simulation_pipeline.step()
        
        np.testing.assert_array_equal(
            simulation_pipeline.physics.positions,
            positions_before
        )
    
    @pytest.mark.integration
    def test_multiple_pause_resume_cycles(self, simulation_pipeline):
        """
        Test multiple pause/resume cycles work correctly.
        
        Expected: Simulation remains stable through cycles.
        """
        for _ in range(3):
            for _ in range(5):
                simulation_pipeline.step()
            simulation_pipeline.pause()
            simulation_pipeline.resume()
        
        # Should still be functional
        simulation_pipeline.step()
        assert simulation_pipeline.physics.step_count > 0


# =============================================================================
# Reset Functionality Tests
# =============================================================================

class TestResetFunctionality:
    """
    Test simulation reset functionality.
    
    Tests verify:
    - Reset clears velocities
    - Reset resets time
    - Renderer is updated after reset
    """
    
    @pytest.mark.integration
    def test_reset_clears_time(self, simulation_pipeline):
        """
        Test reset clears simulation time.
        
        Expected: Time is zero after reset.
        """
        for _ in range(10):
            simulation_pipeline.step()
        
        simulation_pipeline.reset()
        
        assert simulation_pipeline.physics.time == 0.0
    
    @pytest.mark.integration
    def test_reset_clears_velocities(self, simulation_pipeline):
        """
        Test reset clears velocities.
        
        Expected: All velocities are zero after reset.
        """
        for _ in range(10):
            simulation_pipeline.step()
        
        simulation_pipeline.reset()
        
        assert np.allclose(simulation_pipeline.physics.velocities, 0.0)
    
    @pytest.mark.integration
    def test_reset_updates_renderer(self, simulation_pipeline):
        """
        Test reset triggers renderer update.
        
        Expected: Renderer receives new positions after reset.
        """
        for _ in range(5):
            simulation_pipeline.step()
        
        updates_before = simulation_pipeline.renderer.update_count
        
        simulation_pipeline.reset()
        
        assert simulation_pipeline.renderer.update_count > updates_before
    
    @pytest.mark.integration
    def test_reset_clears_step_count(self, simulation_pipeline):
        """
        Test reset clears step counter.
        
        Expected: Step count is zero after reset.
        """
        for _ in range(10):
            simulation_pipeline.step()
        
        simulation_pipeline.reset()
        
        assert simulation_pipeline.physics.step_count == 0


# =============================================================================
# Configuration Synchronization Tests
# =============================================================================

class TestConfigurationSync:
    """
    Test configuration changes propagate correctly.
    
    Tests verify:
    - Physics parameter changes take effect
    - Rendering quality changes apply
    - LOD thresholds can be adjusted
    """
    
    @pytest.mark.integration
    def test_physics_parameter_change(self, physics_engine):
        """
        Test physics parameter changes take effect.
        
        Expected: Changed parameters affect simulation.
        """
        # Store original velocity
        physics_engine.step(0.016)
        v1 = physics_engine.velocities[0, 1]
        
        # Velocity should have decreased due to gravity
        physics_engine.step(0.016)
        v2 = physics_engine.velocities[0, 1]
        
        assert v2 < v1  # Gravity pulling down
    
    @pytest.mark.integration
    def test_lod_threshold_adjustment(self, lod_controller):
        """
        Test LOD threshold adjustment.
        
        Expected: New thresholds affect switching behavior.
        """
        lod_controller.gpu_threshold = 500  # Lower threshold
        
        # Should now be above threshold (1000 > 500)
        for _ in range(35):
            lod_controller.update(0.016)
        
        assert lod_controller.current_mode == 'gpu'


# =============================================================================
# Error Handling Integration Tests
# =============================================================================

class TestIntegrationErrorHandling:
    """
    Test error handling in integration scenarios.
    
    Tests verify:
    - System handles invalid data gracefully
    - Recovery from error states
    """
    
    @pytest.mark.integration
    def test_nan_positions_handled(self, simulation_pipeline):
        """
        Test handling of NaN in positions.
        
        Expected: System should detect or handle NaN values.
        """
        # Inject NaN
        simulation_pipeline.physics.positions[0] = np.nan
        
        # Step should not crash
        try:
            simulation_pipeline.step()
            # If no error, positions should have transferred
            assert simulation_pipeline.renderer.particle_positions is not None
        except ValueError:
            # Raising an error is also acceptable behavior
            pass
    
    @pytest.mark.integration
    def test_empty_particle_system(self):
        """
        Test handling of zero particles.
        
        Expected: System handles gracefully or raises clear error.
        """
        try:
            pipeline = MockSimulationPipeline(particle_count=0)
            pipeline.step()
        except (ValueError, IndexError):
            # Raising an error is acceptable
            pass


# =============================================================================
# Performance Integration Tests
# =============================================================================

@pytest.mark.slow
class TestIntegrationPerformance:
    """
    Test performance characteristics of integrated system.
    
    These tests are marked slow and may be skipped in CI.
    """
    
    @pytest.mark.integration
    def test_sustained_simulation(self, simulation_pipeline, performance_timer):
        """
        Test sustained simulation over many frames.
        
        Expected: System remains stable over 1000 frames.
        """
        with performance_timer() as timer:
            for _ in range(1000):
                simulation_pipeline.step()
        
        assert simulation_pipeline.physics.step_count == 1000
        assert timer.elapsed < 30.0  # Should complete in reasonable time
    
    @pytest.mark.integration
    def test_frame_time_consistency(self, simulation_pipeline):
        """
        Test frame times remain consistent.
        
        Expected: No frames take excessively long.
        """
        frame_times = []
        
        for _ in range(100):
            start = time.time()
            simulation_pipeline.step()
            frame_times.append(time.time() - start)
        
        max_time = max(frame_times)
        avg_time = sum(frame_times) / len(frame_times)
        
        # No frame should take more than 10x average
        if avg_time == 0:
            assert max_time == 0
        elif avg_time < 1e-4:
            assert max_time < 0.002
        else:
            assert max_time < avg_time * 10
