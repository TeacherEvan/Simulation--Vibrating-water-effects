// Interactive Ripple Compute Shader
// Simulates 2D wave equation for water surface disturbance

#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

// Ping-pong buffers for wave simulation
layout(rgba32f, binding = 0) uniform image2D currentHeight;
layout(rgba32f, binding = 1) uniform image2D previousHeight;
layout(rgba32f, binding = 2) uniform image2D velocityMap;  // Output for normal calculation

uniform float deltaTime;
uniform float waveSpeed = 2.0;
uniform float damping = 0.98;
uniform vec2 resolution;

// Interaction points (up to 8 simultaneous touches)
uniform int numInteractions;
uniform vec4 interactions[8];  // xy = position, z = radius, w = strength

void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    
    // Check bounds
    if(pixelCoord.x >= int(resolution.x) || pixelCoord.y >= int(resolution.y)) {
        return;
    }
    
    vec2 uv = vec2(pixelCoord) / resolution;
    
    // Sample current and previous heights
    float current = imageLoad(currentHeight, pixelCoord).r;
    float previous = imageLoad(previousHeight, pixelCoord).r;
    
    // Sample neighbors for Laplacian
    float left   = imageLoad(currentHeight, pixelCoord + ivec2(-1,  0)).r;
    float right  = imageLoad(currentHeight, pixelCoord + ivec2( 1,  0)).r;
    float up     = imageLoad(currentHeight, pixelCoord + ivec2( 0,  1)).r;
    float down   = imageLoad(currentHeight, pixelCoord + ivec2( 0, -1)).r;
    
    // Discrete Laplacian (2D wave equation)
    float laplacian = (left + right + up + down) * 0.25 - current;
    
    // Wave equation integration
    float velocity = (current - previous) * damping + laplacian * waveSpeed * waveSpeed * deltaTime;
    float newHeight = current + velocity * deltaTime;
    
    // Apply interactions
    for(int i = 0; i < numInteractions && i < 8; i++) {
        vec2 interactPos = interactions[i].xy;
        float radius = interactions[i].z;
        float strength = interactions[i].w;
        
        float dist = distance(uv, interactPos);
        if(dist < radius) {
            float influence = 1.0 - smoothstep(0.0, radius, dist);
            newHeight += strength * influence * deltaTime;
        }
    }
    
    // Clamp height
    newHeight = clamp(newHeight, -1.0, 1.0);
    
    // Store new height (swap buffers externally)
    imageStore(previousHeight, pixelCoord, vec4(current, 0.0, 0.0, 0.0));
    imageStore(currentHeight, pixelCoord, vec4(newHeight, 0.0, 0.0, 0.0));
    
    // Calculate velocity for normal map generation
    vec2 gradient = vec2(right - left, up - down) * 0.5;
    float vel = velocity;
    imageStore(velocityMap, pixelCoord, vec4(gradient, vel, 1.0));
}
