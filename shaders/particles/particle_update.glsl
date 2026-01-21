// GPU Particle Update Compute Shader for Panda3D
// Handles physics simulation for foam/spray particles
#version 430
layout(local_size_x = 256) in;

struct Particle {
    vec4 position;   // xyz = position, w = life (0-1)
    vec4 velocity;   // xyz = velocity, w = size
    vec4 color;      // rgba
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

// Simulation parameters
uniform float deltaTime;
uniform vec3 gravity;
uniform vec3 emitterPos;
uniform vec3 emitterVelocity;
uniform float emitterRadius;
uniform float turbulenceStrength;
uniform float drag;
uniform float time;

// Water surface interaction
uniform sampler2D waterHeightMap;
uniform vec2 waterBounds;  // min/max world coordinates

// Random number generation
float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

float hash3(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 45.543))) * 43758.5453);
}

vec3 randomDirection(float seed) {
    float theta = hash(seed) * 6.28318;
    float phi = hash(seed + 1.0) * 3.14159;
    return vec3(
        sin(phi) * cos(theta),
        cos(phi),
        sin(phi) * sin(theta)
    );
}

// Curl noise for turbulence
vec3 curlNoise(vec3 p) {
    float e = 0.01;
    
    float n1 = hash3(p + vec3(e, 0, 0)) - hash3(p - vec3(e, 0, 0));
    float n2 = hash3(p + vec3(0, e, 0)) - hash3(p - vec3(0, e, 0));
    float n3 = hash3(p + vec3(0, 0, e)) - hash3(p - vec3(0, 0, e));
    
    return vec3(n2 - n3, n3 - n1, n1 - n2);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= particles.length()) return;
    
    Particle p = particles[idx];
    
    // Update life
    float lifeDecay = deltaTime * 0.5;  // Particles live ~2 seconds
    p.position.w -= lifeDecay;
    
    if (p.position.w <= 0.0) {
        // ============================================
        // RESPAWN PARTICLE
        // ============================================
        
        float seed = float(idx) + time * 1000.0;
        
        // Spawn around emitter
        vec3 offset = randomDirection(seed) * emitterRadius * hash(seed + 2.0);
        p.position.xyz = emitterPos + offset;
        
        // Random lifetime
        p.position.w = 0.5 + hash(seed + 3.0) * 0.5;
        
        // Initial velocity (upward + emitter velocity + random)
        vec3 randomVel = randomDirection(seed + 4.0);
        p.velocity.xyz = emitterVelocity + 
                         vec3(0, 2.0 + hash(seed + 5.0) * 2.0, 0) +
                         randomVel * 0.5;
        
        // Random size
        p.velocity.w = 0.02 + hash(seed + 6.0) * 0.03;
        
        // Color (white/light blue for foam)
        p.color = vec4(0.9, 0.95, 1.0, 1.0);
        
    } else {
        // ============================================
        // UPDATE PARTICLE PHYSICS
        // ============================================
        
        // Apply gravity
        p.velocity.xyz += gravity * deltaTime;
        
        // Apply drag
        p.velocity.xyz *= pow(drag, deltaTime);
        
        // Turbulence
        vec3 turbulence = curlNoise(p.position.xyz * 2.0 + time) * turbulenceStrength;
        p.velocity.xyz += turbulence * deltaTime;
        
        // Update position
        p.position.xyz += p.velocity.xyz * deltaTime;
        
        // ============================================
        // WATER SURFACE INTERACTION
        // ============================================
        
        // Check if within water bounds
        vec2 waterUV = (p.position.xz - waterBounds.x) / (waterBounds.y - waterBounds.x);
        if (waterUV.x >= 0.0 && waterUV.x <= 1.0 && 
            waterUV.y >= 0.0 && waterUV.y <= 1.0) {
            
            float waterHeight = texture(waterHeightMap, waterUV).r;
            
            // Bounce off water surface
            if (p.position.y < waterHeight && p.velocity.y < 0.0) {
                p.position.y = waterHeight;
                p.velocity.y = -p.velocity.y * 0.3;  // Damped bounce
                p.velocity.xz *= 0.8;  // Friction
                
                // Reduce life on impact
                p.position.w -= 0.1;
            }
        }
        
        // ============================================
        // COLOR/ALPHA FADE
        // ============================================
        
        // Fade out over lifetime
        p.color.a = smoothstep(0.0, 0.3, p.position.w);
        
        // Shrink near end of life
        if (p.position.w < 0.2) {
            p.velocity.w *= 0.95;
        }
    }
    
    particles[idx] = p;
}
