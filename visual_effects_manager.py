"""
Visual Effects Manager for Panda3D Water Simulation
Provides a complete post-processing and visual effects pipeline.

Compatible with Panda3D 1.10+ and NVIDIA A10G GPU
Target: 60 FPS with 10k SPH particles
"""

try:
    from panda3d.core import (
        Shader, Texture, CardMaker, NodePath, Camera,
        GraphicsOutput, FrameBufferProperties, WindowProperties,
        Lens, PerspectiveLens, OrthographicLens,
        LVector2f, LVector3f, LVector4f, LMatrix4f, LPoint3f,
        TextureStage, SamplerState, GeomEnums,
        ComputeNode, ShaderAttrib, ClockObject
    )
    from direct.filter.FilterManager import FilterManager
    from direct.showbase.DirectObject import DirectObject
    from direct.gui.DirectGui import DirectButton
    PANDA3D_AVAILABLE = True
except ImportError:
    # Fallback stubs for testing without Panda3D installed
    PANDA3D_AVAILABLE = False
    
    class DirectButton:
        def __init__(self, *args, **kwargs): pass
        def show(self): pass
        def hide(self): pass

    class DirectObject:
        """Stub class for testing without Panda3D."""
        def accept(self, event, callback):
            pass
    
    class LVector2f:
        def __init__(self, *args): pass
    
    class LVector3f:
        def __init__(self, *args):
            self.x = args[0] if len(args) > 0 else 0.0
            self.y = args[1] if len(args) > 1 else 0.0
            self.z = args[2] if len(args) > 2 else 0.0
        def normalize(self): pass
        def __mul__(self, other): return self
        def __imul__(self, other): return self
    
    class LVector4f:
        def __init__(self, *args): pass
    
    class LMatrix4f:
        def __init__(self, *args): pass
    
    class LPoint3f:
        def __init__(self, *args): pass
    
    class Texture:
        T_float = 0
        F_rgba16 = 0
        F_rgba32 = 0
        F_r32 = 0
        WM_repeat = 0
        FT_nearest = 0
        def __init__(self, name=""): pass
        def setup_2d_texture(self, *args): pass
        def setup_buffer_texture(self, *args): pass
        def setRamImage(self, data): pass
        def setWrapU(self, mode): pass
        def setWrapV(self, mode): pass
        def setMagfilter(self, f): pass
        def setMinfilter(self, f): pass
    
    class Shader:
        SL_GLSL = 0
        @staticmethod
        def load(*args, **kwargs): return None
        @staticmethod
        def load_compute(*args): return None
    
    class GeomEnums:
        UH_dynamic = 0
    
    class ComputeNode:
        def __init__(self, name): pass
        def add_dispatch(self, *args): pass
    
    class FilterManager:
        def __init__(self, win, cam): pass
        def renderSceneInto(self, **kwargs): return MockNodePath()
    
    class MockNodePath:
        def setShader(self, shader): pass
        def setShaderInput(self, name, value): pass
        def attach_new_node(self, node): return self
        def getPos(self): return LVector3f(0, 0, 0)
    
    class ClockObject:
        @staticmethod
        def getGlobalClock():
            return _MockClock()
    
    class _MockClock:
        def getDt(self): return 0.016

import numpy as np
import random
import math


class VisualEffectsManager(DirectObject):
    """
    Main manager for all visual effects in the water simulation.
    Handles post-processing pipeline and effect quality scaling.
    """
    
    def __init__(self, base, config=None):
        """
        Initialize the visual effects manager.
        
        Args:
            base: Panda3D ShowBase instance
            config: Optional configuration dictionary
        """
        super().__init__()
        self.base = base
        
        # Default configuration
        self.config = {
            'bloom_enabled': True,
            'bloom_threshold': 1.0,
            'bloom_intensity': 0.8,
            'bloom_iterations': 5,
            
            'ssao_enabled': True,
            'ssao_samples': 32,
            'ssao_radius': 0.5,
            'ssao_intensity': 1.0,
            
            'ssr_enabled': False,  # Expensive, disable by default
            'ssr_steps': 32,
            'ssr_thickness': 0.1,
            
            'dof_enabled': False,
            'dof_focus_distance': 10.0,
            'dof_focal_length': 50.0,
            
            'tone_mapping': 'aces',  # 'aces', 'reinhard', 'uncharted2'
            'exposure': 1.0,
            'gamma': 2.2,
            
            'target_fps': 60,
            'dynamic_quality': True
        }
        
        if config:
            self.config.update(config)
        
        # Performance tracking
        self.frame_times = []
        self.current_quality_level = 2  # 0=low, 1=medium, 2=high
        
        # Initialize components
        self._init_filter_manager()
        self._init_bloom()
        self._init_ssao()
        self._init_tone_mapping()
        
        # Update task
        self.accept('window-event', self._on_window_resize)
        
    def _init_filter_manager(self):
        """Initialize the Panda3D filter manager for post-processing."""
        self.manager = FilterManager(self.base.win, self.base.cam)
        
        # Create main render textures
        self.scene_texture = Texture("scene")
        self.depth_texture = Texture("depth")
        self.normal_texture = Texture("normal")
        
        # Set up render-to-texture
        self.final_quad = self.manager.renderSceneInto(
            colortex=self.scene_texture,
            depthtex=self.depth_texture,
            auxtex=self.normal_texture
        )
        
    def _init_bloom(self):
        """Initialize bloom post-processing effect."""
        if not self.config['bloom_enabled']:
            self.bloom_textures = []
            return
            
        # Create ping-pong textures for blur
        win_size = (self.base.win.getXSize(), self.base.win.getYSize())
        
        self.bright_texture = Texture("bright")
        self.bloom_textures = []
        
        # Downsampled blur textures (progressive downsample)
        for i in range(self.config['bloom_iterations']):
            scale = 2 ** (i + 1)
            tex = Texture(f"bloom_{i}")
            tex.setup_2d_texture(
                win_size[0] // scale, 
                win_size[1] // scale,
                Texture.T_float, 
                Texture.F_rgba16
            )
            self.bloom_textures.append(tex)
        
        # Load bloom shaders
        self.bright_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/post_process/fullscreen_quad.vert",
            fragment="shaders/post_process/bloom_bright.frag"
        )
        
        self.blur_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/post_process/fullscreen_quad.vert",
            fragment="shaders/post_process/gaussian_blur.frag"
        )
        
    def _init_ssao(self):
        """Initialize SSAO (Screen-Space Ambient Occlusion)."""
        if not self.config['ssao_enabled']:
            return
            
        # Generate sample kernel
        self.ssao_kernel = self._generate_ssao_kernel(self.config['ssao_samples'])
        
        # Generate noise texture (4x4 random rotation vectors)
        self.ssao_noise_texture = self._generate_noise_texture(4, 4)
        
        # SSAO output texture
        self.ssao_texture = Texture("ssao")
        win_size = (self.base.win.getXSize(), self.base.win.getYSize())
        self.ssao_texture.setup_2d_texture(
            win_size[0] // 2,  # Half resolution for performance
            win_size[1] // 2,
            Texture.T_float,
            Texture.F_r32
        )
        
        # Load SSAO shader
        self.ssao_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/post_process/fullscreen_quad.vert",
            fragment="shaders/post_process/ssao.frag"
        )
        
    def _generate_ssao_kernel(self, num_samples):
        """Generate hemisphere sample kernel for SSAO."""
        kernel = []
        
        for i in range(num_samples):
            # Random point in hemisphere
            sample = LVector3f(
                random.random() * 2.0 - 1.0,
                random.random() * 2.0 - 1.0,
                random.random()  # Only positive Z (hemisphere)
            )
            sample.normalize()
            sample *= random.random()
            
            # Scale samples to cluster near origin
            scale = i / num_samples
            scale = 0.1 + scale * scale * 0.9  # lerp(0.1, 1.0, scale^2)
            sample *= scale
            
            kernel.append(sample)
            
        return kernel
    
    def _generate_noise_texture(self, width, height):
        """Generate random rotation vectors for SSAO."""
        noise_data = []
        
        for _ in range(width * height):
            # Random rotation around Z axis
            noise_data.extend([
                random.random() * 2.0 - 1.0,  # x
                random.random() * 2.0 - 1.0,  # y
                0.0,  # z (rotate around surface normal)
                1.0   # padding
            ])
        
        noise_array = np.array(noise_data, dtype=np.float32)
        
        noise_tex = Texture("ssao_noise")
        noise_tex.setup_2d_texture(width, height, Texture.T_float, Texture.F_rgba32)
        noise_tex.setRamImage(noise_array.tobytes())
        noise_tex.setWrapU(Texture.WM_repeat)
        noise_tex.setWrapV(Texture.WM_repeat)
        noise_tex.setMagfilter(Texture.FT_nearest)
        noise_tex.setMinfilter(Texture.FT_nearest)
        
        return noise_tex
        
    def _init_tone_mapping(self):
        """Initialize final tone mapping and composite shader."""
        self.tone_map_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/post_process/fullscreen_quad.vert",
            fragment="shaders/post_process/tone_mapping.frag"
        )
        
        # Apply to final quad
        self.final_quad.setShader(self.tone_map_shader)
        self.final_quad.setShaderInput("hdrScene", self.scene_texture)
        self.final_quad.setShaderInput("exposure", self.config['exposure'])
        self.final_quad.setShaderInput("gamma", self.config['gamma'])
        self.final_quad.setShaderInput("enableBloom", self.config['bloom_enabled'])
        self.final_quad.setShaderInput("enableSSAO", self.config['ssao_enabled'])
        
        if self.config['bloom_enabled'] and self.bloom_textures:
            self.final_quad.setShaderInput("bloomTexture", self.bloom_textures[0])
            self.final_quad.setShaderInput("bloomIntensity", self.config['bloom_intensity'])
            
        if self.config['ssao_enabled']:
            self.final_quad.setShaderInput("ssaoTexture", self.ssao_texture)
    
    def update(self, dt):
        """
        Update visual effects each frame.
        
        Args:
            dt: Delta time in seconds
        """
        # Track frame times for dynamic quality
        if self.config['dynamic_quality']:
            self._update_quality_scaling(dt)
            
    def _update_quality_scaling(self, dt):
        """Dynamically adjust quality based on frame rate."""
        target_frame_time = 1.0 / self.config['target_fps']
        
        self.frame_times.append(dt)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        if avg_frame_time > target_frame_time * 1.2:
            # Too slow, reduce quality
            self._decrease_quality()
        elif avg_frame_time < target_frame_time * 0.8:
            # Headroom available, increase quality
            self._increase_quality()
            
    def _decrease_quality(self):
        """Reduce visual quality for better performance."""
        if self.current_quality_level > 0:
            self.current_quality_level -= 1
            self._apply_quality_level()
            
    def _increase_quality(self):
        """Increase visual quality when performance allows."""
        if self.current_quality_level < 2:
            self.current_quality_level += 1
            self._apply_quality_level()
            
    def _apply_quality_level(self):
        """Apply current quality level settings."""
        quality_presets = {
            0: {  # Low
                'ssao_samples': 16,
                'ssao_radius': 0.3,
                'bloom_iterations': 3,
                'ssr_enabled': False
            },
            1: {  # Medium
                'ssao_samples': 32,
                'ssao_radius': 0.5,
                'bloom_iterations': 4,
                'ssr_enabled': False
            },
            2: {  # High
                'ssao_samples': 64,
                'ssao_radius': 0.8,
                'bloom_iterations': 5,
                'ssr_enabled': True
            }
        }
        
        preset = quality_presets[self.current_quality_level]
        self.config.update(preset)
        
        # Regenerate SSAO kernel if needed
        if self.config['ssao_enabled']:
            self.ssao_kernel = self._generate_ssao_kernel(self.config['ssao_samples'])
            
    def _on_window_resize(self, window):
        """Handle window resize events."""
        # Recreate textures at new resolution
        pass  # Implementation depends on specific needs
        
    def set_exposure(self, value):
        """Set exposure value for tone mapping."""
        self.config['exposure'] = value
        self.final_quad.setShaderInput("exposure", value)
        
    def set_bloom_intensity(self, value):
        """Set bloom intensity."""
        self.config['bloom_intensity'] = value
        if self.config['bloom_enabled']:
            self.final_quad.setShaderInput("bloomIntensity", value)
            
    def toggle_effect(self, effect_name, enabled):
        """Enable or disable a specific effect."""
        key = f"{effect_name}_enabled"
        if key in self.config:
            self.config[key] = enabled
            self.final_quad.setShaderInput(f"enable{effect_name.upper()}", enabled)


class WaterShaderManager:
    """
    Manages water surface rendering with advanced effects.
    """
    
    def __init__(self, base, water_node):
        """
        Initialize water shader manager.
        
        Args:
            base: Panda3D ShowBase instance
            water_node: NodePath of the water surface geometry
        """
        self.base = base
        self.water_node = water_node
        
        # Load water shader
        self.water_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/water/water_surface.vert",
            fragment="shaders/water/water_surface.frag"
        )
        
        # Default water parameters
        self.params = {
            'water_color': LVector3f(0.1, 0.4, 0.6),
            'deep_water_color': LVector3f(0.0, 0.1, 0.3),
            'wave_strength': 0.02,
            'wave_speed': 1.0,
            'shininess': 32.0,
            'foam_threshold': 0.5,
            'max_depth': 5.0
        }
        
        # Gerstner wave parameters
        self.waves = [
            LVector4f(1.0, 0.5, 0.2, 10.0),   # direction.xy, steepness, wavelength
            LVector4f(0.7, 0.3, 0.15, 7.0),
            LVector4f(-0.5, 0.8, 0.1, 5.0),
            LVector4f(0.3, -0.7, 0.08, 3.0)
        ]
        
        self._setup_shader()
        
    def _setup_shader(self):
        """Apply shader and set initial uniforms."""
        self.water_node.setShader(self.water_shader)
        
        # Set wave parameters
        for i, wave in enumerate(self.waves):
            self.water_node.setShaderInput(f"wave{'ABCD'[i]}", wave)
            
        # Set water parameters
        for key, value in self.params.items():
            self.water_node.setShaderInput(key.replace('_', ''), value)
            
        # Set camera/light info
        self.water_node.setShaderInput("cameraPos", self.base.cam.getPos())
        self.water_node.setShaderInput("lightPos", LVector3f(100, 100, 100))
        self.water_node.setShaderInput("lightColor", LVector3f(1.0, 0.95, 0.9))
        
    def update(self, dt, time):
        """Update water shader each frame."""
        self.water_node.setShaderInput("time", time)
        self.water_node.setShaderInput("cameraPos", self.base.cam.getPos())
        
    def set_wave_parameters(self, wave_index, direction, steepness, wavelength):
        """Update a specific wave's parameters."""
        if 0 <= wave_index < 4:
            self.waves[wave_index] = LVector4f(
                direction[0], direction[1], steepness, wavelength
            )
            self.water_node.setShaderInput(
                f"wave{'ABCD'[wave_index]}", 
                self.waves[wave_index]
            )


class GlassSphereShaderManager:
    """
    Manages glass sphere rendering with refraction and interaction effects.
    """
    
    def __init__(self, base, sphere_node):
        """
        Initialize glass sphere shader manager.
        
        Args:
            base: Panda3D ShowBase instance
            sphere_node: NodePath of the glass sphere geometry
        """
        self.base = base
        self.sphere_node = sphere_node
        self.is_broken = False
        
        # Load glass shader
        self.glass_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/glass/glass_refraction.vert",
            fragment="shaders/glass/glass_refraction.frag"
        )
        
        # Glass parameters
        self.params = {
            'IOR': 1.52,  # Index of refraction
            'chromatic_aberration': 0.5,
            'tint_color': LVector3f(0.98, 0.99, 1.0),
            'thickness': 0.5,
            'roughness': 0.05,
            'glow_color': LVector3f(0.5, 0.7, 1.0),
            'glow_intensity': 0.3,
            'glow_power': 3.0,
            'interaction_strength': 0.0
        }
        
        self._setup_ui()
        self._setup_shader()

    def _setup_ui(self):
        """Setup the refreshment UI."""
        if not PANDA3D_AVAILABLE:
            return

        # Create a hidden refresh button
        self.refresh_btn = DirectButton(
            text="Reset Glass",
            scale=0.07,
            pos=(1.1, 0, -0.9),  # Bottom right
            command=self.reset_glass,
            text_fg=(1, 1, 1, 1),
            frameColor=(0.2, 0.2, 0.2, 0.8)
        )
        self.refresh_btn.hide()

    def break_glass(self):
        """Simulate glass breaking and show reset button."""
        if self.is_broken:
            return
            
        self.is_broken = True
        # Visually simulate damage by increasing roughness and interaction glow
        self.params['roughness'] = 0.5 
        self.params['interaction_strength'] = 2.0
        self._setup_shader() # Re-apply params
        
        if hasattr(self, 'refresh_btn'):
            self.refresh_btn.show()
            
    def reset_glass(self):
        """Reset glass state and hide button."""
        self.is_broken = False
        self.params['roughness'] = 0.05
        self.params['interaction_strength'] = 0.0
        self._setup_shader() # Re-apply params
        
        if hasattr(self, 'refresh_btn'):
            self.refresh_btn.hide()

        
    def _setup_shader(self):
        """Apply shader and set initial uniforms."""
        self.sphere_node.setShader(self.glass_shader)
        
        for key, value in self.params.items():
            shader_key = key.replace('_', '')
            if key == 'IOR':
                shader_key = 'IOR'
            self.sphere_node.setShaderInput(shader_key, value)
            
        self.sphere_node.setShaderInput("cameraPos", self.base.cam.getPos())
        
    def update(self, dt, time, interaction_strength=0.0):
        """Update glass shader each frame."""
        self.sphere_node.setShaderInput("time", time)
        self.sphere_node.setShaderInput("cameraPos", self.base.cam.getPos())
        self.sphere_node.setShaderInput("interactionStrength", interaction_strength)
        
    def set_interaction_strength(self, strength):
        """Set the interaction glow strength (from simulation)."""
        self.params['interaction_strength'] = strength
        self.sphere_node.setShaderInput("interactionStrength", strength)


class GPUParticleSystem:
    """
    GPU-accelerated particle system for foam and spray effects.
    """
    
    def __init__(self, base, max_particles=10000):
        """
        Initialize GPU particle system.
        
        Args:
            base: Panda3D ShowBase instance
            max_particles: Maximum number of particles
        """
        self.base = base
        self.max_particles = max_particles
        
        # Create particle data buffer (as texture for compute shader)
        self._init_particle_buffers()
        
        # Load compute shader
        self.update_shader = Shader.load_compute(
            Shader.SL_GLSL,
            "shaders/particles/particle_update.glsl"
        )
        
        # Create compute node
        self._setup_compute_node()
        
        # Simulation parameters
        self.params = {
            'gravity': LVector3f(0, -9.8, 0),
            'emitter_pos': LVector3f(0, 0, 0),
            'emitter_velocity': LVector3f(0, 0, 0),
            'emitter_radius': 0.5,
            'turbulence_strength': 0.5,
            'drag': 0.98
        }
        
    def _init_particle_buffers(self):
        """Initialize particle data storage."""
        # Each particle: position(4) + velocity(4) + color(4) = 12 floats
        particle_data = np.zeros(self.max_particles * 12, dtype=np.float32)
        
        # Initialize with random positions (all dead initially)
        for i in range(self.max_particles):
            base_idx = i * 12
            particle_data[base_idx + 3] = -1.0  # life = -1 (dead)
            
        self.particle_buffer = Texture("particles")
        self.particle_buffer.setup_buffer_texture(
            self.max_particles * 12,
            Texture.T_float,
            Texture.F_rgba32,
            GeomEnums.UH_dynamic
        )
        
    def _setup_compute_node(self):
        """Set up compute shader dispatch."""
        self.compute_node = ComputeNode("particle_update")
        
        # Dispatch enough work groups to cover all particles
        work_groups = (self.max_particles + 255) // 256
        self.compute_node.add_dispatch(work_groups, 1, 1)
        
        self.compute_np = self.base.render.attach_new_node(self.compute_node)
        self.compute_np.setShader(self.update_shader)
        self.compute_np.setShaderInput("particles", self.particle_buffer)
        
    def update(self, dt, time):
        """Update particle simulation."""
        self.compute_np.setShaderInput("deltaTime", dt)
        self.compute_np.setShaderInput("time", time)
        self.compute_np.setShaderInput("gravity", self.params['gravity'])
        self.compute_np.setShaderInput("emitterPos", self.params['emitter_pos'])
        self.compute_np.setShaderInput("emitterVelocity", self.params['emitter_velocity'])
        self.compute_np.setShaderInput("emitterRadius", self.params['emitter_radius'])
        self.compute_np.setShaderInput("turbulenceStrength", self.params['turbulence_strength'])
        self.compute_np.setShaderInput("drag", self.params['drag'])
        
    def set_emitter_position(self, pos):
        """Update emitter position."""
        self.params['emitter_pos'] = LVector3f(*pos)
        
    def set_emitter_velocity(self, vel):
        """Update emitter velocity (for spray effects)."""
        self.params['emitter_velocity'] = LVector3f(*vel)


# Example usage
if __name__ == "__main__":
    if PANDA3D_AVAILABLE:
        from direct.showbase.ShowBase import ShowBase
        
        class WaterSimApp(ShowBase):
            def __init__(self):
                ShowBase.__init__(self)
                
                # Initialize visual effects
                self.vfx = VisualEffectsManager(self, {
                    'bloom_enabled': True,
                    'ssao_enabled': True,
                    'dynamic_quality': True
                })
                
                # Create water surface (example)
                # self.water_node = self.loader.loadModel("models/water_plane")
                # self.water_shader = WaterShaderManager(self, self.water_node)
                
                # Create glass sphere (example)
                # self.sphere_node = self.loader.loadModel("models/glass_sphere")
                # self.glass_shader = GlassSphereShaderManager(self, self.sphere_node)
                
                # Create particle system
                # self.particles = GPUParticleSystem(self, max_particles=10000)
                
                # Update task
                self.time = 0.0
                self.taskMgr.add(self.update, "update")
                
            def update(self, task):
                dt = ClockObject.getGlobalClock().getDt()
                self.time += dt
                
                self.vfx.update(dt)
                # self.water_shader.update(dt, self.time)
                # self.glass_shader.update(dt, self.time)
                # self.particles.update(dt, self.time)
                
                return task.cont
        
        app = WaterSimApp()
        app.run()
    else:
        print("Panda3D is not installed. Install with: pip install panda3d")
