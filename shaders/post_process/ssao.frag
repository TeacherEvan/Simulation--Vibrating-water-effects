// SSAO (Screen-Space Ambient Occlusion) Shader for Panda3D
#version 430

uniform sampler2D depthBuffer;
uniform sampler2D normalBuffer;
uniform sampler2D noiseTexture;  // 4x4 random rotation vectors

uniform mat4 projection;
uniform mat4 projectionInverse;

uniform float radius;
uniform float bias;
uniform float intensity;

// Hemisphere sample kernel (64 samples)
uniform vec3 samples[64];
uniform int kernelSize;

in vec2 texcoord;
out float occlusion;

// Reconstruct view-space position from depth
vec3 reconstructPosition(vec2 uv, float depth) {
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewPos = projectionInverse * clipPos;
    return viewPos.xyz / viewPos.w;
}

void main() {
    float depth = texture(depthBuffer, texcoord).r;
    
    // Skip background pixels
    if (depth >= 1.0) {
        occlusion = 1.0;
        return;
    }
    
    // Reconstruct view-space position
    vec3 position = reconstructPosition(texcoord, depth);
    
    // Get normal (assumed to be in view space already)
    vec3 normal = texture(normalBuffer, texcoord).rgb * 2.0 - 1.0;
    normal = normalize(normal);
    
    // Random rotation from noise texture (tiled 4x4)
    vec2 texSize = vec2(textureSize(depthBuffer, 0));
    vec2 noiseScale = texSize / 4.0;
    vec3 randomVec = texture(noiseTexture, texcoord * noiseScale).xyz * 2.0 - 1.0;
    
    // Create TBN matrix (Gramm-Schmidt process)
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    
    // Sample hemisphere and accumulate occlusion
    float ao = 0.0;
    
    for (int i = 0; i < kernelSize; i++) {
        // Transform sample to view space
        vec3 samplePos = TBN * samples[i];
        samplePos = position + samplePos * radius;
        
        // Project sample to screen space
        vec4 offset = projection * vec4(samplePos, 1.0);
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;
        
        // Sample depth at projected position
        float sampleDepth = texture(depthBuffer, offset.xy).r;
        vec3 samplePosReconstructed = reconstructPosition(offset.xy, sampleDepth);
        
        // Range check to prevent halo artifacts
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(position.z - samplePosReconstructed.z));
        
        // Occlusion test
        ao += (samplePosReconstructed.z >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }
    
    // Normalize and apply intensity
    occlusion = 1.0 - (ao / float(kernelSize)) * intensity;
}
