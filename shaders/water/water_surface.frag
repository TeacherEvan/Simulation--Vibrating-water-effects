// Water Surface Fragment Shader for Panda3D
// Advanced water rendering with Fresnel, reflection, refraction, and foam
#version 430

// Uniforms
uniform sampler2D reflectionMap;
uniform sampler2D refractionMap;
uniform sampler2D normalMap;
uniform sampler2D dudvMap;
uniform sampler2D foamMap;
uniform sampler2D depthMap;
uniform samplerCube environmentMap;

uniform float time;
uniform vec3 cameraPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 waterColor;
uniform vec3 deepWaterColor;

uniform float waveStrength;
uniform float waveSpeed;
uniform float shininess;
uniform float foamThreshold;
uniform float maxDepth;
uniform float near;
uniform float far;

// Inputs from vertex shader
in vec3 worldPos;
in vec4 clipSpace;
in vec2 texCoord;
in vec3 toCamera;
in vec3 fromLight;

// Output
out vec4 fragColor;

// Include common functions
const float PI = 3.14159265359;

float fresnelSchlick(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float linearizeDepth(float depth) {
    return (2.0 * near * far) / (far + near - depth * (far - near));
}

void main() {
    // Animated UV distortion
    vec2 distortedTexCoords = texture(dudvMap, 
        vec2(texCoord.x + time * waveSpeed * 0.03, texCoord.y)).rg * 0.1;
    distortedTexCoords = texCoord + vec2(distortedTexCoords.x, 
        distortedTexCoords.y + time * waveSpeed * 0.03);
    
    vec2 distortion = (texture(dudvMap, distortedTexCoords).rg * 2.0 - 1.0) * waveStrength;
    
    // Screen-space coordinates
    vec2 ndc = (clipSpace.xy / clipSpace.w) * 0.5 + 0.5;
    
    // Apply distortion to reflection/refraction coords
    vec2 reflectCoords = vec2(ndc.x, 1.0 - ndc.y) + distortion;
    vec2 refractCoords = ndc + distortion;
    
    // Clamp to avoid edge artifacts
    reflectCoords = clamp(reflectCoords, 0.001, 0.999);
    refractCoords = clamp(refractCoords, 0.001, 0.999);
    
    // Sample reflection and refraction textures
    vec3 reflection = texture(reflectionMap, reflectCoords).rgb;
    vec3 refraction = texture(refractionMap, refractCoords).rgb;
    
    // Water depth calculation
    float floorDepth = linearizeDepth(texture(depthMap, refractCoords).r);
    float waterDepth = linearizeDepth(gl_FragCoord.z);
    float depth = floorDepth - waterDepth;
    float depthFactor = clamp(depth / maxDepth, 0.0, 1.0);
    
    // Tint refraction based on depth
    vec3 depthColor = mix(waterColor, deepWaterColor, depthFactor);
    refraction = mix(refraction, depthColor, depthFactor * 0.6);
    
    // Normal from normal map (animated)
    vec4 normalMapColor = texture(normalMap, distortedTexCoords);
    vec3 normal = normalize(vec3(
        normalMapColor.r * 2.0 - 1.0,
        normalMapColor.b * 3.0,  // Stronger Y component
        normalMapColor.g * 2.0 - 1.0
    ));
    
    // View direction
    vec3 viewDir = normalize(toCamera);
    
    // Fresnel effect (F0 for water ≈ 0.02)
    float fresnel = fresnelSchlick(max(dot(normal, viewDir), 0.0), 0.02);
    
    // Environment reflection fallback
    vec3 reflectDir = reflect(-viewDir, normal);
    vec3 envReflection = texture(environmentMap, reflectDir).rgb;
    reflection = mix(reflection, envReflection, 0.3);
    
    // Combine reflection and refraction
    vec3 color = mix(refraction, reflection, fresnel);
    
    // Specular highlights (Blinn-Phong)
    vec3 lightDir = normalize(-fromLight);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), shininess);
    vec3 specular = lightColor * spec * 0.5;
    
    // Foam at shallow areas and wave peaks
    float foamFactor = 0.0;
    
    // Depth-based foam (shoreline)
    foamFactor += smoothstep(foamThreshold, 0.0, depth);
    
    // Wave peak foam
    float waveHeight = normalMapColor.b;
    foamFactor += smoothstep(0.6, 0.8, waveHeight) * 0.5;
    
    // Sample foam texture
    vec2 foamUV = worldPos.xz * 0.1 + distortion;
    vec3 foam = texture(foamMap, foamUV).rgb;
    
    // Apply foam
    color = mix(color, foam, clamp(foamFactor, 0.0, 0.8));
    
    // Add specular
    color += specular;
    
    // Soft edges for water meeting geometry
    float alpha = clamp(depth * 2.0, 0.0, 1.0);
    
    fragColor = vec4(color, alpha);
}
