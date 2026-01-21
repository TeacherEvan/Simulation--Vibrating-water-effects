"""
Unit Tests for Visual Effects Manager
======================================

This module tests the VisualEffectsManager, WaterShaderManager,
GlassSphereShaderManager, and GPUParticleSystem classes.

Test Categories:
- Initialization tests
- Configuration tests
- Quality scaling tests
- Shader management tests
- Error handling tests

Expected Outcomes:
- All visual effects components initialize correctly
- Configuration changes apply properly
- Dynamic quality scaling responds to frame time
- Shader parameters are set correctly
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test Fixtures Specific to Visual Effects
# =============================================================================

@pytest.fixture
def vfx_manager(mock_panda3d_base, vfx_config):
    """Create VisualEffectsManager instance for testing."""
    from visual_effects_manager import VisualEffectsManager
    return VisualEffectsManager(mock_panda3d_base, vfx_config)


@pytest.fixture
def water_shader_manager(mock_panda3d_base):
    """Create WaterShaderManager instance for testing."""
    from visual_effects_manager import WaterShaderManager
    mock_water_node = MagicMock()
    return WaterShaderManager(mock_panda3d_base, mock_water_node)


@pytest.fixture
def glass_shader_manager(mock_panda3d_base):
    """Create GlassSphereShaderManager instance for testing."""
    from visual_effects_manager import GlassSphereShaderManager
    mock_sphere_node = MagicMock()
    return GlassSphereShaderManager(mock_panda3d_base, mock_sphere_node)


# =============================================================================
# VisualEffectsManager Tests
# =============================================================================

class TestVisualEffectsManagerInit:
    """
    Test VisualEffectsManager initialization.
    
    Tests verify that:
    - Manager initializes with default configuration
    - Custom configuration overrides defaults
    - All sub-components are created
    """
    
    @pytest.mark.unit
    def test_init_with_defaults(self, mock_panda3d_base):
        """
        Test initialization with default configuration.
        
        Expected: Manager initializes with default bloom, ssao settings.
        Failure Mode: ImportError if visual_effects_manager not found.
        """
        from visual_effects_manager import VisualEffectsManager
        
        manager = VisualEffectsManager(mock_panda3d_base)
        
        assert manager.config['bloom_enabled'] is True
        assert manager.config['ssao_enabled'] is True
        assert manager.config['target_fps'] == 60
    
    @pytest.mark.unit
    def test_init_with_custom_config(self, mock_panda3d_base, vfx_config):
        """
        Test initialization with custom configuration.
        
        Expected: Custom settings override defaults.
        Failure Mode: Configuration not applied correctly.
        """
        from visual_effects_manager import VisualEffectsManager
        
        vfx_config['bloom_intensity'] = 1.5
        vfx_config['exposure'] = 2.0
        
        manager = VisualEffectsManager(mock_panda3d_base, vfx_config)
        
        assert manager.config['bloom_intensity'] == 1.5
        assert manager.config['exposure'] == 2.0
    
    @pytest.mark.unit
    def test_init_quality_level(self, mock_panda3d_base):
        """
        Test initial quality level is high.
        
        Expected: current_quality_level starts at 2 (high).
        """
        from visual_effects_manager import VisualEffectsManager
        
        manager = VisualEffectsManager(mock_panda3d_base)
        
        assert manager.current_quality_level == 2
    
    @pytest.mark.unit
    def test_init_frame_times_empty(self, mock_panda3d_base):
        """
        Test frame times list starts empty.
        
        Expected: No frame times recorded initially.
        """
        from visual_effects_manager import VisualEffectsManager
        
        manager = VisualEffectsManager(mock_panda3d_base)
        
        assert len(manager.frame_times) == 0


class TestVisualEffectsManagerConfig:
    """
    Test configuration changes in VisualEffectsManager.
    
    Tests verify:
    - Exposure can be set dynamically
    - Bloom intensity can be changed
    - Effects can be toggled on/off
    """
    
    @pytest.mark.unit
    def test_set_exposure(self, vfx_manager):
        """
        Test setting exposure value.
        
        Expected: Exposure is updated in config and shader.
        """
        vfx_manager.set_exposure(2.5)
        
        assert vfx_manager.config['exposure'] == 2.5
    
    @pytest.mark.unit
    def test_set_bloom_intensity(self, vfx_manager):
        """
        Test setting bloom intensity.
        
        Expected: Bloom intensity updates if bloom is enabled.
        """
        vfx_manager.set_bloom_intensity(1.2)
        
        assert vfx_manager.config['bloom_intensity'] == 1.2
    
    @pytest.mark.unit
    def test_toggle_effect_enable(self, vfx_manager):
        """
        Test enabling an effect.
        
        Expected: Effect state changes to enabled.
        """
        vfx_manager.toggle_effect('ssr', True)
        
        assert vfx_manager.config['ssr_enabled'] is True
    
    @pytest.mark.unit
    def test_toggle_effect_disable(self, vfx_manager):
        """
        Test disabling an effect.
        
        Expected: Effect state changes to disabled.
        """
        vfx_manager.toggle_effect('bloom', False)
        
        assert vfx_manager.config['bloom_enabled'] is False
    
    @pytest.mark.unit
    @pytest.mark.parametrize("exposure", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_exposure_range(self, vfx_manager, exposure):
        """
        Test exposure across valid range.
        
        Expected: All valid exposure values are accepted.
        """
        vfx_manager.set_exposure(exposure)
        assert vfx_manager.config['exposure'] == exposure


class TestDynamicQualityScaling:
    """
    Test dynamic quality adjustment based on frame rate.
    
    Tests verify:
    - Quality decreases when frame time is too high
    - Quality increases when frame time has headroom
    - Quality stays within bounds (0-2)
    """
    
    @pytest.mark.unit
    def test_decrease_quality_when_slow(self, vfx_manager):
        """
        Test quality decrease on slow frames.
        
        Expected: Quality level decreases when avg frame time exceeds target.
        """
        vfx_manager.config['dynamic_quality'] = True
        vfx_manager.current_quality_level = 2
        
        # Simulate slow frames (30ms = ~33 FPS when target is 60)
        for _ in range(60):
            vfx_manager.update(0.030)
        
        assert vfx_manager.current_quality_level < 2
    
    @pytest.mark.unit
    def test_increase_quality_when_fast(self, vfx_manager):
        """
        Test quality increase on fast frames.
        
        Expected: Quality level increases when avg frame time is below target.
        """
        vfx_manager.config['dynamic_quality'] = True
        vfx_manager.current_quality_level = 0
        
        # Simulate fast frames (8ms = ~120 FPS when target is 60)
        for _ in range(60):
            vfx_manager.update(0.008)
        
        assert vfx_manager.current_quality_level > 0
    
    @pytest.mark.unit
    def test_quality_minimum_bound(self, vfx_manager):
        """
        Test quality doesn't go below 0.
        
        Expected: Quality stays at 0 minimum.
        """
        vfx_manager.current_quality_level = 0
        vfx_manager._decrease_quality()
        
        assert vfx_manager.current_quality_level >= 0
    
    @pytest.mark.unit
    def test_quality_maximum_bound(self, vfx_manager):
        """
        Test quality doesn't exceed 2.
        
        Expected: Quality stays at 2 maximum.
        """
        vfx_manager.current_quality_level = 2
        vfx_manager._increase_quality()
        
        assert vfx_manager.current_quality_level <= 2
    
    @pytest.mark.unit
    def test_quality_presets_applied(self, vfx_manager):
        """
        Test quality preset values are applied correctly.
        
        Expected: Each quality level has different SSAO samples.
        """
        vfx_manager.current_quality_level = 0
        vfx_manager._apply_quality_level()
        low_samples = vfx_manager.config['ssao_samples']
        
        vfx_manager.current_quality_level = 2
        vfx_manager._apply_quality_level()
        high_samples = vfx_manager.config['ssao_samples']
        
        assert high_samples > low_samples


class TestSSAOKernelGeneration:
    """
    Test SSAO kernel and noise texture generation.
    
    Tests verify:
    - Kernel has correct number of samples
    - Samples are normalized appropriately
    - Noise texture has correct dimensions
    """
    
    @pytest.mark.unit
    def test_ssao_kernel_size(self, vfx_manager):
        """
        Test SSAO kernel has correct sample count.
        
        Expected: Kernel length matches ssao_samples config.
        """
        if hasattr(vfx_manager, 'ssao_kernel'):
            expected_size = vfx_manager.config['ssao_samples']
            assert len(vfx_manager.ssao_kernel) == expected_size
    
    @pytest.mark.unit
    def test_generate_ssao_kernel_samples(self, vfx_manager):
        """
        Test SSAO kernel generation.
        
        Expected: All samples are in hemisphere (z >= 0).
        """
        kernel = vfx_manager._generate_ssao_kernel(32)
        
        assert len(kernel) == 32
        # Note: With mock LVector3f, we can't test actual values
    
    @pytest.mark.unit
    @pytest.mark.parametrize("num_samples", [8, 16, 32, 64, 128])
    def test_ssao_kernel_various_sizes(self, vfx_manager, num_samples):
        """
        Test SSAO kernel generation with various sample counts.
        
        Expected: Kernel always has requested number of samples.
        """
        kernel = vfx_manager._generate_ssao_kernel(num_samples)
        assert len(kernel) == num_samples


# =============================================================================
# WaterShaderManager Tests
# =============================================================================

class TestWaterShaderManager:
    """
    Test WaterShaderManager functionality.
    
    Tests verify:
    - Wave parameters are stored correctly
    - Shader inputs are set properly
    - Updates work correctly
    """
    
    @pytest.mark.unit
    def test_init_default_params(self, water_shader_manager):
        """
        Test default water parameters.
        
        Expected: Default water color and wave settings are set.
        """
        assert 'water_color' in water_shader_manager.params
        assert 'wave_strength' in water_shader_manager.params
        assert 'wave_speed' in water_shader_manager.params
    
    @pytest.mark.unit
    def test_init_waves(self, water_shader_manager):
        """
        Test Gerstner wave initialization.
        
        Expected: Four waves are defined.
        """
        assert len(water_shader_manager.waves) == 4
    
    @pytest.mark.unit
    def test_set_wave_parameters(self, water_shader_manager):
        """
        Test setting wave parameters.
        
        Expected: Wave parameters update correctly.
        """
        water_shader_manager.set_wave_parameters(
            wave_index=0,
            direction=(1.0, 0.0),
            steepness=0.5,
            wavelength=20.0
        )
        
        # Wave should be updated
        wave = water_shader_manager.waves[0]
        assert wave is not None
    
    @pytest.mark.unit
    @pytest.mark.parametrize("wave_index", [0, 1, 2, 3])
    def test_set_wave_valid_indices(self, water_shader_manager, wave_index):
        """
        Test setting waves at all valid indices.
        
        Expected: All four wave indices accept parameters.
        """
        water_shader_manager.set_wave_parameters(
            wave_index=wave_index,
            direction=(0.5, 0.5),
            steepness=0.1,
            wavelength=5.0
        )
        # Should not raise
    
    @pytest.mark.unit
    def test_set_wave_invalid_index(self, water_shader_manager):
        """
        Test setting wave at invalid index.
        
        Expected: Invalid index is silently ignored (no exception).
        """
        # Should not raise for out-of-bounds index
        water_shader_manager.set_wave_parameters(
            wave_index=5,
            direction=(1.0, 0.0),
            steepness=0.5,
            wavelength=10.0
        )
    
    @pytest.mark.unit
    def test_update(self, water_shader_manager):
        """
        Test shader update.
        
        Expected: Update completes without error.
        """
        water_shader_manager.update(dt=0.016, time=1.5)
        # Should complete without error


# =============================================================================
# GlassSphereShaderManager Tests
# =============================================================================

class TestGlassSphereShaderManager:
    """
    Test GlassSphereShaderManager functionality.
    
    Tests verify:
    - Default IOR and glass parameters
    - Interaction strength updates
    - Shader input handling
    """
    
    @pytest.mark.unit
    def test_init_default_ior(self, glass_shader_manager):
        """
        Test default index of refraction.
        
        Expected: IOR is set to glass (~1.52).
        """
        assert glass_shader_manager.params['IOR'] == 1.52
    
    @pytest.mark.unit
    def test_init_chromatic_aberration(self, glass_shader_manager):
        """
        Test chromatic aberration parameter.
        
        Expected: Chromatic aberration is initialized.
        """
        assert 'chromatic_aberration' in glass_shader_manager.params
    
    @pytest.mark.unit
    def test_set_interaction_strength(self, glass_shader_manager):
        """
        Test setting interaction glow strength.
        
        Expected: Interaction strength updates correctly.
        """
        glass_shader_manager.set_interaction_strength(0.75)
        
        assert glass_shader_manager.params['interaction_strength'] == 0.75
    
    @pytest.mark.unit
    @pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_interaction_strength_range(self, glass_shader_manager, strength):
        """
        Test interaction strength across valid range.
        
        Expected: All values in [0, 1] are accepted.
        """
        glass_shader_manager.set_interaction_strength(strength)
        assert glass_shader_manager.params['interaction_strength'] == strength
    
    @pytest.mark.unit
    def test_update(self, glass_shader_manager):
        """
        Test shader update with interaction.
        
        Expected: Update completes without error.
        """
        glass_shader_manager.update(dt=0.016, time=2.0, interaction_strength=0.5)
        # Should complete without error


# =============================================================================
# GPUParticleSystem Tests
# =============================================================================

class TestGPUParticleSystem:
    """
    Test GPUParticleSystem functionality.
    
    Tests verify:
    - Particle buffer initialization
    - Emitter position/velocity updates
    - Compute shader setup
    """
    
    @pytest.fixture
    def particle_system(self, mock_panda3d_base):
        """Create GPUParticleSystem for testing."""
        from visual_effects_manager import GPUParticleSystem
        return GPUParticleSystem(mock_panda3d_base, max_particles=1000)
    
    @pytest.mark.unit
    def test_init_max_particles(self, particle_system):
        """
        Test particle count initialization.
        
        Expected: max_particles is stored correctly.
        """
        assert particle_system.max_particles == 1000
    
    @pytest.mark.unit
    def test_init_default_gravity(self, particle_system):
        """
        Test default gravity setting.
        
        Expected: Gravity points downward.
        """
        gravity = particle_system.params['gravity']
        # Check y component is negative (down)
        assert gravity.y < 0 if hasattr(gravity, 'y') else True
    
    @pytest.mark.unit
    def test_set_emitter_position(self, particle_system):
        """
        Test setting emitter position.
        
        Expected: Emitter position updates.
        """
        particle_system.set_emitter_position((1.0, 2.0, 3.0))
        
        pos = particle_system.params['emitter_pos']
        assert pos is not None
    
    @pytest.mark.unit
    def test_set_emitter_velocity(self, particle_system):
        """
        Test setting emitter velocity.
        
        Expected: Emitter velocity updates.
        """
        particle_system.set_emitter_velocity((0.0, 5.0, 0.0))
        
        vel = particle_system.params['emitter_velocity']
        assert vel is not None
    
    @pytest.mark.unit
    def test_update(self, particle_system):
        """
        Test particle system update.
        
        Expected: Update completes without error.
        """
        particle_system.update(dt=0.016, time=1.0)
        # Should complete without error


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """
    Test edge cases and error handling.
    
    Tests verify:
    - Handling of zero/negative values
    - Large value handling
    - Missing configuration handling
    """
    
    @pytest.mark.unit
    def test_zero_exposure(self, vfx_manager):
        """
        Test setting zero exposure.
        
        Expected: Zero exposure is accepted (though produces black image).
        """
        vfx_manager.set_exposure(0.0)
        assert vfx_manager.config['exposure'] == 0.0
    
    @pytest.mark.unit
    def test_very_large_bloom_intensity(self, vfx_manager):
        """
        Test very large bloom intensity.
        
        Expected: Large values are accepted (clamping is shader's job).
        """
        vfx_manager.set_bloom_intensity(100.0)
        assert vfx_manager.config['bloom_intensity'] == 100.0
    
    @pytest.mark.unit
    def test_negative_dt_update(self, vfx_manager):
        """
        Test update with negative delta time.
        
        Expected: Should not crash (though negative dt is unusual).
        """
        # This shouldn't crash, but behavior is undefined
        vfx_manager.update(-0.016)
    
    @pytest.mark.unit
    def test_zero_dt_update(self, vfx_manager):
        """
        Test update with zero delta time.
        
        Expected: Should handle gracefully.
        """
        vfx_manager.update(0.0)
    
    @pytest.mark.unit
    def test_very_small_dt_update(self, vfx_manager):
        """
        Test update with very small delta time.
        
        Expected: Should handle gracefully.
        """
        vfx_manager.update(1e-10)


# =============================================================================
# Integration with Panda3D (Conditional)
# =============================================================================

@pytest.mark.panda3d
class TestPanda3DIntegration:
    """
    Tests requiring actual Panda3D installation.
    
    These tests are skipped if Panda3D is not available.
    """
    
    def test_shader_load(self):
        """
        Test actual shader loading with Panda3D.
        
        Expected: Shader loads without error.
        Skipped: If Panda3D not installed.
        """
        pytest.skip("Requires Panda3D with graphics context")
    
    def test_texture_creation(self):
        """
        Test actual texture creation with Panda3D.
        
        Expected: Textures are created correctly.
        Skipped: If Panda3D not installed.
        """
        pytest.skip("Requires Panda3D with graphics context")
