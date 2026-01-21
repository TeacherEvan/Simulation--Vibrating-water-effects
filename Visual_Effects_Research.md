# Advanced Visual Effects & Shader Techniques for Panda3D Water Simulation

## Executive Summary

This document provides comprehensive research on implementing advanced visual effects for a Python-based SPH water simulation using Panda3D, targeting NVIDIA A10G GPU (24GB VRAM, Ampere architecture) at 60 FPS with 10k particles.

**Performance Budget Analysis (60 FPS = 16.67ms per frame)**:

- SPH Simulation: ~4-5ms (compute shader)
- G-Buffer Pass: ~1-2ms
- Post-Processing Pipeline: ~4-6ms
- Visual Effects: ~3-4ms
- Remaining Headroom: ~2-3ms

---

## 1. Water Visual Effects

### 1.1 Screen-Space Reflections (SSR)

SSR creates realistic water surface reflections by ray-marching through the depth buffer.

**Performance Cost**: 1.5-3ms depending on quality settings

**GLSL Implementation for Panda3D**:

```glsl
#version 430

uniform sampler2D colorBuffer;
uniform sampler2D depthBuffer;
uniform sampler2D normalBuffer;
uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ViewMatrix;
uniform float maxDistance;
uniform float resolution;
uniform float thickness;

in vec2 texcoord;
out vec4 fragColor;

vec3 SSR(vec3 position, vec3 reflection, vec3 color) {
    vec3 step = reflection * 0.1;
    vec3 marchingPosition = position + step;
    float delta;
    float depthFromScreen;
    vec2 screenPosition;

    for (int i = 0; i < 30; i++) {
        // Project to screen space
        vec4 projected = p3d_ProjectionMatrix * vec4(marchingPosition, 1.0);
        screenPosition = projected.xy / projected.w * 0.5 + 0.5;

        // Check bounds
        if (screenPosition.x < 0.0 || screenPosition.x > 1.0 ||
            screenPosition.y < 0.0 || screenPosition.y > 1.0) {
            return color;
        }

        // Sample depth and compare
        depthFromScreen = texture(depthBuffer, screenPosition).r;
        delta = marchingPosition.z - depthFromScreen;

        if (abs(delta) < thickness) {
            // Binary search refinement
            vec3 hitColor = texture(colorBuffer, screenPosition).rgb;
            float fresnel = pow(1.0 - dot(reflection, vec3(0, 1, 0)), 3.0);
            return mix(color, hitColor, fresnel * 0.5);
        }

        marchingPosition += step;
        if (length(marchingPosition - position) > maxDistance) break;
    }

    return color;
}

void main() {
    vec3 normal = texture(normalBuffer, texcoord).rgb * 2.0 - 1.0;
    vec3 position = /* reconstruct from depth */;
    vec3 viewDir = normalize(-position);
    vec3 reflection = reflect(-viewDir, normal);

    vec3 color = texture(colorBuffer, texcoord).rgb;
    fragColor = vec4(SSR(position, reflection, color), 1.0);
}
```

**Panda3D Integration**:

```python
from panda3d.core import Shader, Texture, CardMaker
from direct.filter.FilterManager import FilterManager

class SSREffect:
    def __init__(self, base):
        self.manager = FilterManager(base.win, base.cam)

        # Create render targets
        self.color_tex = Texture()
        self.depth_tex = Texture()
        self.normal_tex = Texture()

        # Setup render-to-texture
        self.quad = self.manager.renderSceneInto(
            colortex=self.color_tex,
            depthtex=self.depth_tex,
            auxtex=self.normal_tex
        )

        # Load SSR shader
        self.ssr_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/fullscreen_quad.vert",
            fragment="shaders/ssr.frag"
        )

        self.quad.setShader(self.ssr_shader)
        self.quad.setShaderInput("colorBuffer", self.color_tex)
        self.quad.setShaderInput("depthBuffer", self.depth_tex)
        self.quad.setShaderInput("normalBuffer", self.normal_tex)
        self.quad.setShaderInput("maxDistance", 50.0)
        self.quad.setShaderInput("thickness", 0.1)
```

---

### 1.2 Caustics Rendering

Caustics create light patterns refracted through water onto underwater surfaces.

**Approach 1: Texture-Based Caustics (Fast, ~0.3ms)**

```glsl
#version 430

uniform sampler2D causticTexture;
uniform float time;
uniform vec3 lightDir;

vec3 applyCaustics(vec3 worldPos, vec3 baseColor, float waterDepth) {
    // Animated UV coordinates
    vec2 causticUV1 = worldPos.xz * 0.1 + time * 0.05;
    vec2 causticUV2 = worldPos.xz * 0.15 - time * 0.03;

    // Sample and blend two caustic layers
    float caustic1 = texture(causticTexture, causticUV1).r;
    float caustic2 = texture(causticTexture, causticUV2).r;
    float causticIntensity = min(caustic1 + caustic2, 1.5);

    // Attenuate by depth
    float depthAttenuation = exp(-waterDepth * 0.5);
    causticIntensity *= depthAttenuation;

    // Apply directional light influence
    float lightInfluence = max(dot(vec3(0, 1, 0), lightDir), 0.0);

    return baseColor + vec3(causticIntensity * lightInfluence * 0.4);
}
```

**Approach 2: Ray-Traced Caustics (High Quality, ~1-2ms with compute)**

```glsl
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D waterSurface;    // Water height/normal
uniform writeonly image2D causticMap;
uniform vec3 lightDir;
uniform float IOR;  // Index of refraction for water: 1.33

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(coord) / vec2(imageSize(causticMap));

    // Get water surface normal
    vec3 normal = texture(waterSurface, uv).rgb;

    // Refract light through surface
    vec3 refractedDir = refract(-lightDir, normal, 1.0 / IOR);

    // Project to floor
    float floorY = -1.0;  // Floor height
    float waterY = texture(waterSurface, uv).a;
    float t = (floorY - waterY) / refractedDir.y;
    vec2 hitPos = uv + refractedDir.xz * t * 0.1;

    // Accumulate light intensity (simplified)
    float intensity = 1.0 / (1.0 + t * 0.1);

    imageStore(causticMap, ivec2(hitPos * vec2(imageSize(causticMap))),
               vec4(intensity, intensity, intensity, 1.0));
}
```

---

### 1.3 Fresnel Effect Implementation

The Fresnel effect controls reflection/refraction ratio based on viewing angle.

**Schlick's Approximation (Industry Standard)**:

```glsl
// Fresnel-Schlick approximation
float fresnelSchlick(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Full implementation for water shader
vec3 calculateWaterColor(vec3 viewDir, vec3 normal, vec3 reflectionColor,
                         vec3 refractionColor, float roughness) {
    // Water F0 is approximately 0.02 (from IOR 1.33)
    float F0 = 0.02;

    // Calculate view angle
    float NdotV = max(dot(normal, viewDir), 0.0);

    // Fresnel term
    float fresnel = fresnelSchlick(NdotV, F0);

    // Roughness affects fresnel spread
    fresnel = mix(fresnel, 1.0, roughness * 0.5);

    // Blend reflection and refraction
    return mix(refractionColor, reflectionColor, fresnel);
}
```

**Water Surface Shader with Fresnel**:

```glsl
#version 430

uniform sampler2D reflectionMap;
uniform sampler2D refractionMap;
uniform sampler2D normalMap;
uniform sampler2D dudvMap;  // Distortion map
uniform float time;
uniform vec3 cameraPos;
uniform vec3 waterColor;
uniform float waveStrength;

in vec3 worldPos;
in vec4 clipSpace;
out vec4 fragColor;

void main() {
    // Animated distortion
    vec2 distortedTexCoords = texture(dudvMap,
        vec2(worldPos.x + time * 0.03, worldPos.z) * 0.1).rg * 0.1;
    distortedTexCoords += vec2(worldPos.x, worldPos.z + time * 0.03) * 0.1;
    vec2 distortion = (texture(dudvMap, distortedTexCoords).rg * 2.0 - 1.0) * waveStrength;

    // Project to screen space
    vec2 ndc = (clipSpace.xy / clipSpace.w) * 0.5 + 0.5;
    vec2 reflectCoords = vec2(ndc.x, 1.0 - ndc.y) + distortion;
    vec2 refractCoords = ndc + distortion;

    // Clamp to avoid edge artifacts
    reflectCoords = clamp(reflectCoords, 0.001, 0.999);
    refractCoords = clamp(refractCoords, 0.001, 0.999);

    // Sample textures
    vec3 reflection = texture(reflectionMap, reflectCoords).rgb;
    vec3 refraction = texture(refractionMap, refractCoords).rgb;
    refraction = mix(refraction, waterColor, 0.3);  // Tint

    // Normal from normal map
    vec3 normal = texture(normalMap, distortedTexCoords).rgb;
    normal = normalize(vec3(normal.r * 2.0 - 1.0, normal.b, normal.g * 2.0 - 1.0));

    // Fresnel
    vec3 viewDir = normalize(cameraPos - worldPos);
    float fresnel = fresnelSchlick(max(dot(normal, viewDir), 0.0), 0.02);

    fragColor = vec4(mix(refraction, reflection, fresnel), 1.0);
}
```

---

### 1.4 Subsurface Scattering for Water

SSS simulates light penetrating and scattering within water volume.

**Performance Cost**: ~0.5-1ms

```glsl
#version 430

// Simplified screen-space SSS for water
uniform sampler2D depthBuffer;
uniform sampler2D colorBuffer;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform float sssStrength;
uniform float sssRadius;
uniform vec3 waterAbsorption;  // RGB absorption coefficients

vec3 calculateSSS(vec3 worldPos, vec3 normal, float depth) {
    // Light penetration based on thickness
    vec3 lightDir = normalize(lightPos - worldPos);

    // Thickness estimation (simplified)
    float thickness = depth * 2.0;

    // Absorption based on thickness
    vec3 absorption = exp(-thickness * waterAbsorption);

    // Wrap lighting for soft SSS effect
    float NdotL = dot(normal, lightDir);
    float wrap = (NdotL + 0.5) / 1.5;
    wrap = max(wrap, 0.0);

    // SSS contribution
    vec3 sss = lightColor * absorption * wrap * sssStrength;

    // View-dependent transmission
    vec3 viewDir = normalize(-worldPos);
    float VdotL = max(dot(viewDir, -lightDir), 0.0);
    float transmission = pow(VdotL, 4.0) * thickness;

    return sss + lightColor * transmission * absorption * 0.2;
}
```

---

### 1.5 Dynamic Ripple/Wave Shaders

**Gerstner Waves (Physically-Based)**:

```glsl
#version 430

struct GerstnerWave {
    vec2 direction;
    float steepness;
    float wavelength;
    float speed;
};

uniform GerstnerWave waves[4];
uniform float time;

vec3 gerstnerWave(vec2 position, GerstnerWave wave, inout vec3 tangent, inout vec3 binormal) {
    float k = 2.0 * 3.14159 / wave.wavelength;
    float c = sqrt(9.8 / k);
    vec2 d = normalize(wave.direction);
    float f = k * (dot(d, position) - c * time * wave.speed);
    float a = wave.steepness / k;

    tangent += vec3(
        -d.x * d.x * wave.steepness * sin(f),
        d.x * wave.steepness * cos(f),
        -d.x * d.y * wave.steepness * sin(f)
    );

    binormal += vec3(
        -d.x * d.y * wave.steepness * sin(f),
        d.y * wave.steepness * cos(f),
        -d.y * d.y * wave.steepness * sin(f)
    );

    return vec3(
        d.x * a * cos(f),
        a * sin(f),
        d.y * a * cos(f)
    );
}

vec3 calculateWavePosition(vec2 basePos, out vec3 normal) {
    vec3 tangent = vec3(1, 0, 0);
    vec3 binormal = vec3(0, 0, 1);
    vec3 displacement = vec3(0);

    for (int i = 0; i < 4; i++) {
        displacement += gerstnerWave(basePos, waves[i], tangent, binormal);
    }

    normal = normalize(cross(binormal, tangent));
    return vec3(basePos.x, 0, basePos.y) + displacement;
}
```

**Interactive Ripples (GPU-based)**:

```glsl
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D heightMap;
layout(rgba32f, binding = 1) uniform image2D velocityMap;

uniform float damping;
uniform float tension;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(heightMap);

    if (coord.x >= size.x || coord.y >= size.y) return;

    // Sample neighbors
    float center = imageLoad(heightMap, coord).r;
    float left = imageLoad(heightMap, coord + ivec2(-1, 0)).r;
    float right = imageLoad(heightMap, coord + ivec2(1, 0)).r;
    float up = imageLoad(heightMap, coord + ivec2(0, -1)).r;
    float down = imageLoad(heightMap, coord + ivec2(0, 1)).r;

    // Wave equation
    float velocity = imageLoad(velocityMap, coord).r;
    float acceleration = tension * (left + right + up + down - 4.0 * center);
    velocity = (velocity + acceleration) * damping;
    float newHeight = center + velocity;

    // Calculate normal
    vec3 normal = normalize(vec3(left - right, 2.0, up - down));

    imageStore(heightMap, coord, vec4(newHeight, normal));
    imageStore(velocityMap, coord, vec4(velocity, 0, 0, 0));
}
```

---

### 1.6 Foam and Spray Particle Effects

**GPU Instanced Foam Particles**:

```glsl
#version 430

// Vertex shader for instanced foam sprites
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ViewMatrix;
uniform sampler2D particlePositions;  // Instance data texture
uniform float particleSize;
uniform float time;

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
in int gl_InstanceID;

out vec2 texCoord;
out float alpha;
out float life;

void main() {
    // Fetch particle data from texture
    ivec2 texSize = textureSize(particlePositions, 0);
    ivec2 texCoord2D = ivec2(gl_InstanceID % texSize.x, gl_InstanceID / texSize.x);
    vec4 particleData = texelFetch(particlePositions, texCoord2D, 0);

    vec3 position = particleData.xyz;
    life = particleData.w;

    // Billboard facing camera
    vec3 cameraRight = vec3(p3d_ViewMatrix[0][0], p3d_ViewMatrix[1][0], p3d_ViewMatrix[2][0]);
    vec3 cameraUp = vec3(p3d_ViewMatrix[0][1], p3d_ViewMatrix[1][1], p3d_ViewMatrix[2][1]);

    // Scale based on life
    float scale = particleSize * (1.0 - life * 0.5);
    vec3 vertexPos = position
        + cameraRight * p3d_Vertex.x * scale
        + cameraUp * p3d_Vertex.y * scale;

    gl_Position = p3d_ModelViewProjectionMatrix * vec4(vertexPos, 1.0);
    texCoord = p3d_MultiTexCoord0;
    alpha = 1.0 - life;  // Fade out over lifetime
}
```

**Foam Fragment Shader**:

```glsl
#version 430

uniform sampler2D foamTexture;
uniform sampler2D noiseTexture;
uniform float time;

in vec2 texCoord;
in float alpha;
in float life;

out vec4 fragColor;

void main() {
    // Animated foam texture
    vec2 animatedUV = texCoord + vec2(sin(time + life * 6.28), cos(time * 0.7)) * 0.1;
    vec4 foam = texture(foamTexture, animatedUV);

    // Noise-based dissolution
    float noise = texture(noiseTexture, texCoord * 2.0).r;
    float dissolve = step(life, noise);

    // Soft edges
    float softEdge = 1.0 - smoothstep(0.4, 0.5, length(texCoord - 0.5));

    fragColor = vec4(foam.rgb, foam.a * alpha * dissolve * softEdge);
}
```

---

## 2. Glass Sphere Effects

### 2.1 Refraction Shaders with IOR

**Performance Cost**: ~0.5-1ms per sphere

```glsl
#version 430

uniform samplerCube environmentMap;
uniform sampler2D sceneTexture;
uniform mat4 p3d_ViewMatrixInverse;
uniform float IOR;  // Glass: 1.5, Water: 1.33
uniform float chromaticAberration;
uniform vec3 tintColor;
uniform float thickness;

in vec3 worldNormal;
in vec3 worldPos;
in vec3 viewDir;
in vec4 clipPos;

out vec4 fragColor;

void main() {
    vec3 N = normalize(worldNormal);
    vec3 V = normalize(viewDir);

    // Calculate refraction with chromatic aberration
    float etaR = 1.0 / (IOR - chromaticAberration * 0.02);
    float etaG = 1.0 / IOR;
    float etaB = 1.0 / (IOR + chromaticAberration * 0.02);

    vec3 refractR = refract(-V, N, etaR);
    vec3 refractG = refract(-V, N, etaG);
    vec3 refractB = refract(-V, N, etaB);

    // Environment sampling with refraction
    vec3 refractionColor;
    refractionColor.r = texture(environmentMap, refractR).r;
    refractionColor.g = texture(environmentMap, refractG).g;
    refractionColor.b = texture(environmentMap, refractB).b;

    // Reflection
    vec3 R = reflect(-V, N);
    vec3 reflectionColor = texture(environmentMap, R).rgb;

    // Fresnel (glass F0 ≈ 0.04)
    float fresnel = fresnelSchlick(max(dot(N, V), 0.0), 0.04);

    // Apply thickness-based absorption
    vec3 absorption = exp(-thickness * (1.0 - tintColor));
    refractionColor *= absorption;

    // Combine
    vec3 finalColor = mix(refractionColor, reflectionColor, fresnel);

    fragColor = vec4(finalColor, 1.0);
}
```

### 2.2 Surface Deformation Visualization

For visualizing vibration-induced deformation on the glass sphere:

```glsl
#version 430

uniform float deformationAmount;
uniform float frequency;
uniform float time;
uniform sampler2D deformationMap;  // From SPH simulation

in vec3 localPos;
in vec3 localNormal;
out vec4 p3d_FragData[2];  // MRT for deferred

vec3 applyDeformation(vec3 pos, vec3 normal, out vec3 newNormal) {
    // Sample displacement from simulation
    vec2 sphereUV = vec2(
        atan(pos.z, pos.x) / (2.0 * 3.14159) + 0.5,
        acos(pos.y / length(pos)) / 3.14159
    );

    float displacement = texture(deformationMap, sphereUV).r * deformationAmount;

    // Add procedural vibration pattern
    float vibration = sin(frequency * time + pos.y * 10.0) * 0.01;
    displacement += vibration;

    // Deform along normal
    vec3 deformedPos = pos + normal * displacement;

    // Calculate new normal (finite differences)
    float epsilon = 0.001;
    vec3 dx = applyDeformationBase(pos + vec3(epsilon, 0, 0)) -
              applyDeformationBase(pos - vec3(epsilon, 0, 0));
    vec3 dz = applyDeformationBase(pos + vec3(0, 0, epsilon)) -
              applyDeformationBase(pos - vec3(0, 0, epsilon));
    newNormal = normalize(cross(dx, dz));

    return deformedPos;
}
```

### 2.3 Interaction Glow Effects (Rim Lighting)

```glsl
#version 430

uniform vec3 glowColor;
uniform float glowIntensity;
uniform float glowPower;
uniform float interactionStrength;  // From simulation

in vec3 worldNormal;
in vec3 viewDir;

vec3 calculateRimGlow(vec3 baseColor) {
    float NdotV = max(dot(normalize(worldNormal), normalize(viewDir)), 0.0);

    // Rim factor (stronger at edges)
    float rim = 1.0 - NdotV;
    rim = pow(rim, glowPower);

    // Modulate by interaction strength
    float dynamicGlow = rim * glowIntensity * (1.0 + interactionStrength * 2.0);

    // Animated pulse based on interaction
    float pulse = 1.0 + sin(time * 10.0 * interactionStrength) * 0.2 * interactionStrength;
    dynamicGlow *= pulse;

    return baseColor + glowColor * dynamicGlow;
}
```

### 2.4 Environmental Reflection Mapping

```glsl
#version 430

uniform samplerCube environmentCube;
uniform sampler2D roughnessMap;
uniform float envIntensity;
uniform int maxMipLevel;

vec3 sampleEnvironmentLOD(vec3 reflectDir, float roughness) {
    // Calculate LOD based on roughness
    float lod = roughness * float(maxMipLevel);

    // Sample with LOD
    vec3 envColor = textureLod(environmentCube, reflectDir, lod).rgb;

    return envColor * envIntensity;
}

// Box-Projected Cube Map for accurate reflections
vec3 boxProjectedCubemap(vec3 reflectDir, vec3 worldPos,
                         vec3 boxMin, vec3 boxMax, vec3 boxCenter) {
    vec3 firstPlaneIntersect = (boxMax - worldPos) / reflectDir;
    vec3 secondPlaneIntersect = (boxMin - worldPos) / reflectDir;
    vec3 furthestPlane = max(firstPlaneIntersect, secondPlaneIntersect);
    float dist = min(min(furthestPlane.x, furthestPlane.y), furthestPlane.z);

    vec3 intersectPos = worldPos + reflectDir * dist;
    return intersectPos - boxCenter;
}
```

---

## 3. Post-Processing Pipeline

### 3.1 Recommended Pipeline Order

```
┌─────────────────────────────────────────────────────────────┐
│                    RENDERING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│ 1. G-Buffer Pass (Deferred)            │ ~1-2ms             │
│    - Position, Normal, Albedo, Depth   │                    │
├────────────────────────────────────────┼────────────────────┤
│ 2. Lighting Pass                       │ ~1-2ms             │
│    - Direct lighting                   │                    │
│    - SSR (Screen-Space Reflections)    │                    │
├────────────────────────────────────────┼────────────────────┤
│ 3. Volumetric Effects                  │ ~1-2ms             │
│    - God rays / Light shafts           │                    │
│    - Underwater fog                    │                    │
├────────────────────────────────────────┼────────────────────┤
│ 4. Transparent Pass (Forward)          │ ~1-2ms             │
│    - Water surface                     │                    │
│    - Particles (foam, spray)           │                    │
├────────────────────────────────────────┼────────────────────┤
│ 5. Post-Processing Chain               │ ~2-4ms             │
│    a. SSAO                             │                    │
│    b. Bloom                            │                    │
│    c. DOF (optional)                   │                    │
│    d. Motion Blur (optional)           │                    │
│    e. Chromatic Aberration             │                    │
│    f. TAA                              │                    │
│    g. Tone Mapping + Gamma             │                    │
└────────────────────────────────────────┴────────────────────┘
```

### 3.2 Bloom Implementation

**Performance Cost**: ~0.8-1.5ms (depends on iterations)

```glsl
// Brightness extraction shader
#version 430

uniform sampler2D hdrScene;
uniform float threshold;
uniform float softThreshold;

in vec2 texcoord;
out vec4 brightColor;

void main() {
    vec3 color = texture(hdrScene, texcoord).rgb;

    // Luminance-based extraction
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));

    // Soft knee threshold
    float soft = brightness - threshold + softThreshold;
    soft = clamp(soft, 0.0, 2.0 * softThreshold);
    soft = soft * soft / (4.0 * softThreshold + 0.00001);

    float contribution = max(soft, brightness - threshold);
    contribution /= max(brightness, 0.00001);

    brightColor = vec4(color * contribution, 1.0);
}
```

**Gaussian Blur (Separable)**:

```glsl
#version 430

uniform sampler2D inputTexture;
uniform bool horizontal;
uniform float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

in vec2 texcoord;
out vec4 fragColor;

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(inputTexture, 0));
    vec3 result = texture(inputTexture, texcoord).rgb * weights[0];

    vec2 offset = horizontal ? vec2(texelSize.x, 0.0) : vec2(0.0, texelSize.y);

    for (int i = 1; i < 5; i++) {
        result += texture(inputTexture, texcoord + offset * float(i)).rgb * weights[i];
        result += texture(inputTexture, texcoord - offset * float(i)).rgb * weights[i];
    }

    fragColor = vec4(result, 1.0);
}
```

**Panda3D Bloom Manager**:

```python
class BloomEffect:
    def __init__(self, base, threshold=1.0, intensity=0.8, blur_iterations=5):
        self.base = base
        self.threshold = threshold
        self.intensity = intensity

        # Setup filter manager
        self.manager = FilterManager(base.win, base.cam)

        # Create textures
        self.scene_tex = Texture()
        self.bright_tex = Texture()
        self.blur_texA = Texture()
        self.blur_texB = Texture()

        # Main scene render
        self.quad = self.manager.renderSceneInto(colortex=self.scene_tex)

        # Brightness extraction pass
        self._setup_bright_pass()

        # Blur passes (ping-pong)
        self._setup_blur_passes(blur_iterations)

        # Final composite
        self._setup_composite()

    def _setup_bright_pass(self):
        bright_shader = Shader.load(Shader.SL_GLSL,
            vertex="shaders/fullscreen.vert",
            fragment="shaders/bloom_bright.frag")
        # ... setup bright extraction

    def _setup_blur_passes(self, iterations):
        blur_shader = Shader.load(Shader.SL_GLSL,
            vertex="shaders/fullscreen.vert",
            fragment="shaders/gaussian_blur.frag")
        # ... setup ping-pong blur
```

### 3.3 Depth of Field (Bokeh)

**Performance Cost**: ~1-2ms

```glsl
#version 430

uniform sampler2D colorBuffer;
uniform sampler2D depthBuffer;
uniform float focusDistance;
uniform float focusRange;
uniform float bokehRadius;
uniform float nearBlurScale;
uniform float farBlurScale;

const int SAMPLES = 16;
const vec2 poissonDisk[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2(0.94558609, -0.76890725),
    vec2(-0.094184101, -0.92938870),
    vec2(0.34495938, 0.29387760),
    // ... more samples
);

in vec2 texcoord;
out vec4 fragColor;

float linearizeDepth(float d) {
    float near = 0.1;
    float far = 1000.0;
    return (2.0 * near * far) / (far + near - d * (far - near));
}

void main() {
    float depth = linearizeDepth(texture(depthBuffer, texcoord).r);

    // Calculate blur amount based on distance from focus
    float blur = 0.0;
    if (depth < focusDistance) {
        blur = (focusDistance - depth) / focusRange * nearBlurScale;
    } else {
        blur = (depth - focusDistance) / focusRange * farBlurScale;
    }
    blur = clamp(blur, 0.0, 1.0) * bokehRadius;

    // Bokeh sampling
    vec3 color = vec3(0.0);
    float totalWeight = 0.0;

    for (int i = 0; i < SAMPLES; i++) {
        vec2 offset = poissonDisk[i] * blur / vec2(textureSize(colorBuffer, 0));
        float sampleDepth = linearizeDepth(texture(depthBuffer, texcoord + offset).r);

        // Weight based on depth similarity (prevents bleeding)
        float weight = 1.0 / (1.0 + abs(depth - sampleDepth) * 0.1);

        color += texture(colorBuffer, texcoord + offset).rgb * weight;
        totalWeight += weight;
    }

    fragColor = vec4(color / totalWeight, 1.0);
}
```

### 3.4 Volumetric Lighting / God Rays

**Performance Cost**: ~1-2ms

```glsl
#version 430

uniform sampler2D depthBuffer;
uniform sampler2D sceneColor;
uniform vec3 lightScreenPos;  // Light position in screen space
uniform float exposure;
uniform float decay;
uniform float density;
uniform float weight;
uniform int numSamples;

in vec2 texcoord;
out vec4 fragColor;

void main() {
    vec2 deltaTexCoord = (texcoord - lightScreenPos.xy);
    deltaTexCoord *= 1.0 / float(numSamples) * density;

    vec2 samplePos = texcoord;
    vec3 godRays = vec3(0.0);
    float illuminationDecay = 1.0;

    for (int i = 0; i < numSamples; i++) {
        samplePos -= deltaTexCoord;

        // Only accumulate if not occluded
        float depth = texture(depthBuffer, samplePos).r;
        float occluded = step(0.99, depth);  // Sky check

        vec3 sampleColor = texture(sceneColor, samplePos).rgb * occluded;
        sampleColor *= illuminationDecay * weight;
        godRays += sampleColor;

        illuminationDecay *= decay;
    }

    vec3 original = texture(sceneColor, texcoord).rgb;
    fragColor = vec4(original + godRays * exposure, 1.0);
}
```

### 3.5 Chromatic Aberration

**Performance Cost**: ~0.1-0.2ms

```glsl
#version 430

uniform sampler2D inputTexture;
uniform float strength;
uniform vec2 center;  // Usually (0.5, 0.5)

in vec2 texcoord;
out vec4 fragColor;

void main() {
    vec2 direction = texcoord - center;
    float dist = length(direction);

    // Offset increases toward edges
    vec2 offset = direction * dist * strength;

    // Sample each channel with offset
    float r = texture(inputTexture, texcoord + offset).r;
    float g = texture(inputTexture, texcoord).g;
    float b = texture(inputTexture, texcoord - offset).b;

    fragColor = vec4(r, g, b, 1.0);
}
```

### 3.6 Motion Blur for Fast Particles

**Per-Object Motion Blur**:

```glsl
#version 430

uniform sampler2D colorBuffer;
uniform sampler2D velocityBuffer;  // Screen-space velocity
uniform float blurScale;
uniform int maxSamples;

in vec2 texcoord;
out vec4 fragColor;

void main() {
    vec2 velocity = texture(velocityBuffer, texcoord).rg * blurScale;

    // Clamp velocity to prevent artifacts
    float speed = length(velocity);
    if (speed > 0.1) velocity = velocity / speed * 0.1;

    vec3 color = vec3(0.0);
    int samples = max(1, int(speed * float(maxSamples)));

    for (int i = 0; i < samples; i++) {
        float t = float(i) / float(samples - 1) - 0.5;
        vec2 offset = velocity * t;
        color += texture(colorBuffer, texcoord + offset).rgb;
    }

    fragColor = vec4(color / float(samples), 1.0);
}
```

### 3.7 SSAO (Screen-Space Ambient Occlusion)

**Performance Cost**: ~1-2ms

```glsl
#version 430

uniform sampler2D depthBuffer;
uniform sampler2D normalBuffer;
uniform sampler2D noiseTexture;  // 4x4 random vectors
uniform vec3 samples[64];         // Hemisphere samples
uniform mat4 projection;
uniform float radius;
uniform float bias;
uniform float intensity;

in vec2 texcoord;
out float occlusion;

vec3 reconstructPosition(vec2 uv, float depth) {
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewPos = inverse(projection) * clipPos;
    return viewPos.xyz / viewPos.w;
}

void main() {
    float depth = texture(depthBuffer, texcoord).r;
    if (depth >= 1.0) {
        occlusion = 1.0;
        return;
    }

    vec3 position = reconstructPosition(texcoord, depth);
    vec3 normal = texture(normalBuffer, texcoord).rgb * 2.0 - 1.0;

    // Random rotation from noise texture
    vec2 noiseScale = vec2(textureSize(depthBuffer, 0)) / 4.0;
    vec3 randomVec = texture(noiseTexture, texcoord * noiseScale).xyz * 2.0 - 1.0;

    // Create TBN matrix
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    // Sample and accumulate occlusion
    float ao = 0.0;
    for (int i = 0; i < 64; i++) {
        vec3 samplePos = TBN * samples[i];
        samplePos = position + samplePos * radius;

        // Project sample
        vec4 offset = projection * vec4(samplePos, 1.0);
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = texture(depthBuffer, offset.xy).r;
        vec3 samplePosWorld = reconstructPosition(offset.xy, sampleDepth);

        // Range check and occlusion test
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(position.z - samplePosWorld.z));
        ao += (samplePosWorld.z >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion = 1.0 - (ao / 64.0) * intensity;
}
```

---

## 4. Particle Systems

### 4.1 GPU Instanced Particles

**Performance**: Can handle 100k+ particles at 60 FPS

```python
# Panda3D GPU Instanced Particle System
from panda3d.core import (
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode, Texture, Shader
)
import numpy as np

class GPUParticleSystem:
    def __init__(self, base, max_particles=10000):
        self.base = base
        self.max_particles = max_particles

        # Create instance data texture (position, velocity, life)
        self.instance_tex = Texture()
        self.instance_tex.setup_2d_texture(
            max_particles, 1,
            Texture.T_float, Texture.F_rgba32
        )

        # Create billboard quad geometry
        self._create_quad_geometry()

        # Compute shader for particle update
        self.update_shader = Shader.load_compute(
            Shader.SL_GLSL, "shaders/particle_update.glsl"
        )

        # Render shader
        self.render_shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/particle_instanced.vert",
            fragment="shaders/particle.frag"
        )

    def update(self, dt):
        # Dispatch compute shader for physics
        self.base.graphicsEngine.dispatch_compute(
            (self.max_particles // 256, 1, 1),
            self.update_attrib,
            self.base.win.get_gsg()
        )
```

**Compute Shader for Particle Physics**:

```glsl
#version 430
layout(local_size_x = 256) in;

struct Particle {
    vec4 position;  // xyz = pos, w = life
    vec4 velocity;  // xyz = vel, w = size
    vec4 color;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

uniform float deltaTime;
uniform vec3 gravity;
uniform vec3 emitterPos;
uniform float turbulence;
uniform float time;

// Simple noise function
float noise(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.543))) * 43758.5453);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= particles.length()) return;

    Particle p = particles[idx];

    // Update life
    p.position.w -= deltaTime;

    if (p.position.w <= 0.0) {
        // Respawn particle
        p.position.xyz = emitterPos + vec3(
            noise(vec3(idx, time, 0)) - 0.5,
            noise(vec3(idx, time, 1)) - 0.5,
            noise(vec3(idx, time, 2)) - 0.5
        ) * 0.5;
        p.position.w = 1.0 + noise(vec3(idx, time, 3)) * 0.5;
        p.velocity.xyz = vec3(
            noise(vec3(idx, time, 4)) - 0.5,
            noise(vec3(idx, time, 5)) * 2.0,
            noise(vec3(idx, time, 6)) - 0.5
        );
    } else {
        // Apply physics
        p.velocity.xyz += gravity * deltaTime;

        // Turbulence
        vec3 turb = vec3(
            noise(p.position.xyz * 10.0 + time),
            noise(p.position.xyz * 10.0 + time + 100.0),
            noise(p.position.xyz * 10.0 + time + 200.0)
        ) - 0.5;
        p.velocity.xyz += turb * turbulence * deltaTime;

        // Update position
        p.position.xyz += p.velocity.xyz * deltaTime;

        // Fade color
        p.color.a = smoothstep(0.0, 0.2, p.position.w);
    }

    particles[idx] = p;
}
```

### 4.2 Sprite-Based vs Mesh Particles

| Aspect         | Sprite (Billboard) | Mesh              |
| -------------- | ------------------ | ----------------- |
| Performance    | ~0.001ms per 1000  | ~0.01ms per 1000  |
| Visual Quality | Good for glow/soft | Better for debris |
| Memory         | Low (4 vertices)   | Higher (varies)   |
| Use Case       | Foam, spray, mist  | Droplets, chunks  |

**Recommendation**: Use sprites for foam/mist (95% of particles), mesh for large droplets.

### 4.3 Trail Renderers for Fast Particles

```glsl
#version 430

// Trail vertex shader
struct TrailPoint {
    vec3 position;
    float width;
    float age;
};

layout(std430, binding = 0) buffer TrailBuffer {
    TrailPoint trail[];
};

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform vec3 cameraPos;
uniform float maxAge;

out float vAge;
out vec2 vUV;

void main() {
    uint pointIdx = gl_VertexID / 2;
    uint side = gl_VertexID % 2;

    TrailPoint point = trail[pointIdx];
    TrailPoint nextPoint = trail[min(pointIdx + 1, trail.length() - 1)];

    // Calculate perpendicular direction
    vec3 dir = normalize(nextPoint.position - point.position);
    vec3 toCamera = normalize(cameraPos - point.position);
    vec3 side_vec = normalize(cross(dir, toCamera));

    // Width based on age
    float width = point.width * (1.0 - point.age / maxAge);

    vec3 offset = side_vec * width * (side == 0 ? 1.0 : -1.0);
    vec3 worldPos = point.position + offset;

    gl_Position = p3d_ModelViewProjectionMatrix * vec4(worldPos, 1.0);
    vAge = point.age / maxAge;
    vUV = vec2(float(pointIdx) / float(trail.length()), float(side));
}
```

---

## 5. Performance Optimization

### 5.1 Deferred vs Forward Rendering Trade-offs

| Aspect             | Deferred           | Forward                 |
| ------------------ | ------------------ | ----------------------- |
| Many Lights        | ✅ Excellent       | ❌ Poor                 |
| Transparency       | ❌ Requires hybrid | ✅ Native               |
| MSAA               | ❌ Expensive       | ✅ Native               |
| Memory             | Higher (G-Buffer)  | Lower                   |
| Overdraw           | ✅ Minimal impact  | ❌ Expensive            |
| **Recommendation** | Use for scene      | Use for water/particles |

**Hybrid Approach for Water Simulation**:

```
1. Deferred Pass: Scene geometry (glass sphere base, environment)
2. Forward Pass: Water surface, particles (transparency needed)
3. Post-Process: Apply effects to combined output
```

### 5.2 LOD for Visual Effects

```python
class EffectLODManager:
    """Dynamic quality scaling based on GPU load"""

    def __init__(self, target_fps=60):
        self.target_frame_time = 1000.0 / target_fps
        self.quality_levels = {
            'ssao': {'samples': [16, 32, 64], 'radius': [0.3, 0.5, 1.0]},
            'ssr': {'steps': [16, 32, 64], 'resolution': [0.25, 0.5, 1.0]},
            'bloom': {'iterations': [2, 4, 6]},
            'dof': {'samples': [8, 16, 32]},
            'particles': {'max_count': [2000, 5000, 10000]},
        }
        self.current_level = 1  # Medium

    def update(self, frame_time_ms):
        """Adjust quality based on frame time"""
        if frame_time_ms > self.target_frame_time * 1.2:
            # Too slow, reduce quality
            self.current_level = max(0, self.current_level - 1)
        elif frame_time_ms < self.target_frame_time * 0.8:
            # Headroom available, increase quality
            self.current_level = min(2, self.current_level + 1)

        return self._get_current_settings()

    def _get_current_settings(self):
        settings = {}
        for effect, params in self.quality_levels.items():
            settings[effect] = {
                k: v[self.current_level] for k, v in params.items()
            }
        return settings
```

### 5.3 Temporal Anti-Aliasing (TAA)

**Performance Cost**: ~0.5-1ms

```glsl
#version 430

uniform sampler2D currentFrame;
uniform sampler2D historyBuffer;
uniform sampler2D velocityBuffer;
uniform sampler2D depthBuffer;
uniform vec2 jitterOffset;
uniform float blendFactor;

in vec2 texcoord;
out vec4 fragColor;

vec3 RGBToYCoCg(vec3 rgb) {
    return vec3(
        0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b,
        0.5 * rgb.r - 0.5 * rgb.b,
        -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b
    );
}

vec3 YCoCgToRGB(vec3 ycocg) {
    return vec3(
        ycocg.x + ycocg.y - ycocg.z,
        ycocg.x + ycocg.z,
        ycocg.x - ycocg.y - ycocg.z
    );
}

void main() {
    // Get velocity for reprojection
    vec2 velocity = texture(velocityBuffer, texcoord).rg;
    vec2 historyCoord = texcoord - velocity;

    // Sample current and history
    vec3 current = texture(currentFrame, texcoord).rgb;
    vec3 history = texture(historyBuffer, historyCoord).rgb;

    // Neighborhood clamping in YCoCg space
    vec3 minColor = vec3(1e10);
    vec3 maxColor = vec3(-1e10);

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = vec2(x, y) / vec2(textureSize(currentFrame, 0));
            vec3 sample = RGBToYCoCg(texture(currentFrame, texcoord + offset).rgb);
            minColor = min(minColor, sample);
            maxColor = max(maxColor, sample);
        }
    }

    // Clamp history to neighborhood
    vec3 historyYCoCg = RGBToYCoCg(history);
    historyYCoCg = clamp(historyYCoCg, minColor, maxColor);
    history = YCoCgToRGB(historyYCoCg);

    // Blend
    float blend = blendFactor;

    // Reduce blend for disoccluded pixels
    if (historyCoord.x < 0.0 || historyCoord.x > 1.0 ||
        historyCoord.y < 0.0 || historyCoord.y > 1.0) {
        blend = 1.0;
    }

    fragColor = vec4(mix(history, current, blend), 1.0);
}
```

**Jitter Pattern for TAA**:

```python
class TAAJitter:
    """Halton sequence jitter for TAA"""

    def __init__(self, sample_count=16):
        self.sample_count = sample_count
        self.frame_index = 0
        self.jitter_offsets = self._generate_halton_sequence()

    def _halton(self, index, base):
        result = 0.0
        f = 1.0
        while index > 0:
            f /= base
            result += f * (index % base)
            index //= base
        return result

    def _generate_halton_sequence(self):
        offsets = []
        for i in range(self.sample_count):
            x = self._halton(i + 1, 2) - 0.5
            y = self._halton(i + 1, 3) - 0.5
            offsets.append((x, y))
        return offsets

    def get_current_jitter(self, viewport_size):
        offset = self.jitter_offsets[self.frame_index % self.sample_count]
        # Convert to pixel offset
        return (
            offset[0] / viewport_size[0],
            offset[1] / viewport_size[1]
        )

    def advance_frame(self):
        self.frame_index += 1
```

### 5.4 Compute Shader Optimizations

**Shared Memory Usage**:

```glsl
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

shared vec4 sharedTile[18][18];  // 16x16 + 1 pixel border

uniform sampler2D inputTexture;
uniform writeonly image2D outputTexture;

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 lid = ivec2(gl_LocalInvocationID.xy);
    ivec2 groupId = ivec2(gl_WorkGroupID.xy);

    // Cooperative loading into shared memory
    ivec2 baseCoord = groupId * 16 - 1;

    // Each thread loads one or more pixels
    if (lid.x < 18 && lid.y < 18) {
        ivec2 loadCoord = baseCoord + lid;
        sharedTile[lid.y][lid.x] = texelFetch(inputTexture, loadCoord, 0);
    }

    // Handle border loading
    if (lid.x < 2 && lid.y < 18) {
        ivec2 loadCoord = baseCoord + ivec2(lid.x + 16, lid.y);
        sharedTile[lid.y][lid.x + 16] = texelFetch(inputTexture, loadCoord, 0);
    }
    // ... similar for y border

    barrier();
    memoryBarrierShared();

    // Now process using shared memory (much faster than texture reads)
    vec4 result = vec4(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            result += sharedTile[lid.y + 1 + y][lid.x + 1 + x] * weight;
        }
    }

    imageStore(outputTexture, gid, result);
}
```

---

## 6. Implementation Priority Matrix

### Phase 1: Essential Effects (Week 1-2)

| Effect                     | Performance Cost | Visual Impact | Priority     |
| -------------------------- | ---------------- | ------------- | ------------ |
| Fresnel + Basic Reflection | 0.3ms            | ⭐⭐⭐⭐⭐    | **CRITICAL** |
| Bloom                      | 0.8ms            | ⭐⭐⭐⭐      | HIGH         |
| Basic Water Shader         | 0.5ms            | ⭐⭐⭐⭐⭐    | **CRITICAL** |
| Glass Refraction           | 0.5ms            | ⭐⭐⭐⭐      | HIGH         |

### Phase 2: Enhancement Effects (Week 3-4)

| Effect             | Performance Cost | Visual Impact | Priority |
| ------------------ | ---------------- | ------------- | -------- |
| SSR                | 1.5ms            | ⭐⭐⭐⭐      | MEDIUM   |
| SSAO               | 1.0ms            | ⭐⭐⭐        | MEDIUM   |
| Caustics (Texture) | 0.3ms            | ⭐⭐⭐⭐      | MEDIUM   |
| Foam Particles     | 0.5ms            | ⭐⭐⭐        | MEDIUM   |

### Phase 3: Polish Effects (Week 5-6)

| Effect               | Performance Cost | Visual Impact | Priority |
| -------------------- | ---------------- | ------------- | -------- |
| TAA                  | 0.8ms            | ⭐⭐⭐        | LOW      |
| DOF                  | 1.0ms            | ⭐⭐⭐        | LOW      |
| God Rays             | 1.0ms            | ⭐⭐⭐        | LOW      |
| Motion Blur          | 0.5ms            | ⭐⭐          | LOW      |
| Chromatic Aberration | 0.1ms            | ⭐⭐          | LOW      |

---

## 7. Performance Benchmarks (A10G Estimates)

| Configuration            | Estimated FPS | Notes                 |
| ------------------------ | ------------- | --------------------- |
| Minimal (Essential only) | 120+          | Development/debugging |
| Medium (Phase 1+2)       | 75-90         | Good quality          |
| Full (All effects)       | 55-65         | Maximum quality       |
| Full + Dynamic LOD       | 60 stable     | **Recommended**       |

### Memory Budget (24GB VRAM)

| Resource               | Memory   |
| ---------------------- | -------- |
| G-Buffer (1080p)       | ~100MB   |
| Post-Process Buffers   | ~200MB   |
| Particle Data (10k)    | ~10MB    |
| Environment Maps       | ~50MB    |
| Textures               | ~200MB   |
| SPH Simulation         | ~500MB   |
| **Total Used**         | **~1GB** |
| **Available Headroom** | **23GB** |

---

## 8. Reference Shader Code Repository Structure

```
shaders/
├── water/
│   ├── water_surface.vert
│   ├── water_surface.frag
│   ├── caustics.glsl
│   ├── ripples_compute.glsl
│   └── foam_particle.glsl
├── glass/
│   ├── glass_refraction.vert
│   ├── glass_refraction.frag
│   ├── rim_glow.glsl
│   └── env_mapping.glsl
├── post_process/
│   ├── fullscreen_quad.vert
│   ├── bloom_bright.frag
│   ├── gaussian_blur.frag
│   ├── ssr.frag
│   ├── ssao.frag
│   ├── dof_bokeh.frag
│   ├── god_rays.frag
│   ├── taa.frag
│   ├── chromatic_aberration.frag
│   └── tone_mapping.frag
├── particles/
│   ├── particle_update.glsl (compute)
│   ├── particle_render.vert
│   ├── particle_render.frag
│   └── trail_render.glsl
└── common/
    ├── noise.glsl
    ├── pbr_functions.glsl
    ├── depth_utils.glsl
    └── color_utils.glsl
```

---

## 9. Next Steps

1. **Create shader infrastructure** in Panda3D with FilterManager
2. **Implement essential water shader** with Fresnel reflection
3. **Add bloom post-processing** for glowing effects
4. **Integrate glass sphere** with refraction
5. **Add dynamic LOD system** for stable 60 FPS
6. **Progressive enhancement** with SSR, SSAO, caustics
7. **Polish pass** with TAA, DOF, volumetric effects

This research provides the complete technical foundation for implementing professional-quality visual effects in the water simulation project.
