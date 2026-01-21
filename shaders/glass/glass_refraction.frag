// Glass Sphere Refraction Fragment Shader for Panda3D
// Features: IOR-based refraction, chromatic aberration, Fresnel, rim glow
#version 430

// Uniforms
uniform samplerCube environmentMap;
uniform sampler2D sceneTexture;

uniform mat4 p3d_ViewMatrixInverse;
uniform vec3 cameraPos;

uniform float IOR;                    // Index of refraction (glass: 1.52)
uniform float chromaticAberration;    // Chromatic aberration strength
uniform vec3 tintColor;               // Glass tint
uniform float thickness;              // Material thickness for absorption
uniform float roughness;              // Surface roughness

uniform vec3 glowColor;               // Rim glow color
uniform float glowIntensity;          // Rim glow intensity
uniform float glowPower;              // Rim glow falloff

uniform float interactionStrength;    // Dynamic interaction from simulation
uniform float time;

// Inputs
in vec3 worldNormal;
in vec3 worldPos;
in vec3 viewDir;
in vec4 clipPos;
in vec2 texCoord;

// Output
out vec4 fragColor;

// Fresnel approximation
float fresnelSchlick(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 N = normalize(worldNormal);
    vec3 V = normalize(viewDir);
    
    // Calculate eta values for RGB channels (chromatic aberration)
    float etaR = 1.0 / (IOR - chromaticAberration * 0.02);
    float etaG = 1.0 / IOR;
    float etaB = 1.0 / (IOR + chromaticAberration * 0.02);
    
    // Calculate refraction directions for each channel
    vec3 refractR = refract(-V, N, etaR);
    vec3 refractG = refract(-V, N, etaG);
    vec3 refractB = refract(-V, N, etaB);
    
    // Handle total internal reflection
    bool tirR = length(refractR) < 0.01;
    bool tirG = length(refractG) < 0.01;
    bool tirB = length(refractB) < 0.01;
    
    // Sample environment with refraction
    vec3 refractionColor;
    refractionColor.r = tirR ? 0.0 : texture(environmentMap, refractR).r;
    refractionColor.g = tirG ? 0.0 : texture(environmentMap, refractG).g;
    refractionColor.b = tirB ? 0.0 : texture(environmentMap, refractB).b;
    
    // Reflection
    vec3 R = reflect(-V, N);
    
    // Sample environment for reflection with LOD based on roughness
    float lod = roughness * 8.0;  // Assuming 8 mip levels
    vec3 reflectionColor = textureLod(environmentMap, R, lod).rgb;
    
    // Fresnel term (glass F0 ≈ 0.04)
    float F0 = 0.04;
    float fresnel = fresnelSchlick(max(dot(N, V), 0.0), F0);
    
    // Handle total internal reflection - increase reflection
    if (tirR || tirG || tirB) {
        fresnel = mix(fresnel, 1.0, 0.5);
    }
    
    // Apply thickness-based absorption (Beer's law)
    vec3 absorption = exp(-thickness * (1.0 - tintColor));
    refractionColor *= absorption;
    
    // Combine reflection and refraction
    vec3 color = mix(refractionColor, reflectionColor, fresnel);
    
    // ============================================
    // RIM GLOW / INTERACTION EFFECT
    // ============================================
    
    float NdotV = max(dot(N, V), 0.0);
    
    // Rim factor (stronger at edges)
    float rim = 1.0 - NdotV;
    rim = pow(rim, glowPower);
    
    // Modulate by interaction strength from simulation
    float dynamicGlow = rim * glowIntensity * (1.0 + interactionStrength * 3.0);
    
    // Animated pulse based on interaction
    float pulse = 1.0 + sin(time * 10.0 * interactionStrength) * 0.3 * interactionStrength;
    dynamicGlow *= pulse;
    
    // Apply rim glow
    color += glowColor * dynamicGlow;
    
    // ============================================
    // SURFACE VARIATION (subtle)
    // ============================================
    
    // Add subtle surface variation based on position
    float surfaceNoise = sin(worldPos.x * 20.0 + time) * sin(worldPos.y * 20.0 + time * 0.7);
    surfaceNoise = surfaceNoise * 0.02 + 1.0;
    color *= surfaceNoise;
    
    fragColor = vec4(color, 1.0);
}
