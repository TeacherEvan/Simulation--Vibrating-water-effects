// PBR Common Functions for Panda3D Water Simulation
// Compatible with GLSL 430

#ifndef PBR_FUNCTIONS_GLSL
#define PBR_FUNCTIONS_GLSL

const float PI = 3.14159265359;

// ============================================
// FRESNEL FUNCTIONS
// ============================================

// Schlick's approximation for Fresnel
float fresnelSchlick(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickVec3(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fresnel with roughness (for IBL)
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ============================================
// NORMAL DISTRIBUTION FUNCTIONS (NDF)
// ============================================

// GGX/Trowbridge-Reitz NDF
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return num / denom;
}

// ============================================
// GEOMETRY FUNCTIONS
// ============================================

// Schlick-GGX geometry function
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return num / denom;
}

// Smith's method combining view and light geometry
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// ============================================
// REFRACTION
// ============================================

// Calculate refracted direction with chromatic aberration
vec3 refractChromatic(vec3 I, vec3 N, float etaR, float etaG, float etaB, out vec3 refractR, out vec3 refractG, out vec3 refractB) {
    refractR = refract(I, N, etaR);
    refractG = refract(I, N, etaG);
    refractB = refract(I, N, etaB);
    return refractG; // Return center wavelength as primary
}

// Index of refraction for common materials
const float IOR_WATER = 1.333;
const float IOR_GLASS = 1.52;
const float IOR_AIR = 1.0;

// F0 calculation from IOR
float calculateF0(float ior) {
    float f = (ior - 1.0) / (ior + 1.0);
    return f * f;
}

// ============================================
// DEPTH UTILITIES
// ============================================

// Linearize depth from NDC
float linearizeDepth(float depth, float near, float far) {
    return (2.0 * near * far) / (far + near - depth * (far - near));
}

// Reconstruct view-space position from depth
vec3 reconstructPosition(vec2 uv, float depth, mat4 invProjection) {
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewPos = invProjection * clipPos;
    return viewPos.xyz / viewPos.w;
}

// ============================================
// COLOR UTILITIES
// ============================================

// sRGB to linear
vec3 sRGBToLinear(vec3 srgb) {
    return pow(srgb, vec3(2.2));
}

// Linear to sRGB
vec3 linearToSRGB(vec3 linear) {
    return pow(linear, vec3(1.0 / 2.2));
}

// Luminance calculation (ITU BT.709)
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// ACES tone mapping
vec3 acesToneMap(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Reinhard tone mapping
vec3 reinhardToneMap(vec3 color) {
    return color / (color + vec3(1.0));
}

// ============================================
// NOISE FUNCTIONS
// ============================================

// Simple hash function
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float hash3D(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.543))) * 43758.5453);
}

// Value noise
float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// FBM (Fractal Brownian Motion)
float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < octaves; i++) {
        value += amplitude * valueNoise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

// ============================================
// WAVE FUNCTIONS
// ============================================

// Single Gerstner wave
vec3 gerstnerWave(vec2 position, vec2 direction, float steepness, float wavelength, float speed, float time, inout vec3 tangent, inout vec3 binormal) {
    float k = 2.0 * PI / wavelength;
    float c = sqrt(9.8 / k);
    vec2 d = normalize(direction);
    float f = k * (dot(d, position) - c * time * speed);
    float a = steepness / k;
    
    tangent += vec3(
        -d.x * d.x * steepness * sin(f),
        d.x * steepness * cos(f),
        -d.x * d.y * steepness * sin(f)
    );
    
    binormal += vec3(
        -d.x * d.y * steepness * sin(f),
        d.y * steepness * cos(f),
        -d.y * d.y * steepness * sin(f)
    );
    
    return vec3(
        d.x * a * cos(f),
        a * sin(f),
        d.y * a * cos(f)
    );
}

#endif // PBR_FUNCTIONS_GLSL
