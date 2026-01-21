// Caustics Shader for underwater light patterns
// Uses noise-based caustics with animated movement
// Can be applied to underwater surfaces or projected

#version 430 core

in vec2 texCoords;
in vec3 worldPos;
in vec3 worldNormal;

out vec4 FragColor;

uniform sampler2D causticsTexture;  // Optional: precomputed caustics
uniform sampler2D normalMap;

uniform float time;
uniform vec3 lightDir;
uniform vec3 lightColor = vec3(1.0, 0.95, 0.9);
uniform float causticsScale = 2.0;
uniform float causticsSpeed = 0.5;
uniform float causticsIntensity = 0.5;
uniform float waterSurfaceY = 0.0;
uniform float maxDepth = 10.0;

// Simplex noise functions
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x * 34.0) + 1.0) * x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    
    return 130.0 * dot(m, g);
}

// Fractional Brownian Motion for caustics pattern
float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for(int i = 0; i < octaves; i++) {
        value += amplitude * snoise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    
    return value;
}

// Voronoi-based caustics for more realistic look
float voronoiCaustics(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float minDist1 = 10.0;
    float minDist2 = 10.0;
    
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = vec2(
                snoise(i + neighbor + vec2(time * causticsSpeed * 0.7, 0.0)),
                snoise(i + neighbor + vec2(0.0, time * causticsSpeed * 0.7) + 100.0)
            ) * 0.5 + 0.5;
            
            vec2 diff = neighbor + point - f;
            float dist = length(diff);
            
            if(dist < minDist1) {
                minDist2 = minDist1;
                minDist1 = dist;
            } else if(dist < minDist2) {
                minDist2 = dist;
            }
        }
    }
    
    return minDist2 - minDist1;
}

// Calculate caustics intensity at a point
vec3 calculateCaustics(vec3 position) {
    // Project onto water surface plane
    float depth = waterSurfaceY - position.y;
    
    // Skip above water or too deep
    if(depth <= 0.0 || depth > maxDepth) {
        return vec3(0.0);
    }
    
    // Depth attenuation
    float depthFade = 1.0 - smoothstep(0.0, maxDepth, depth);
    
    // Sample caustics at multiple scales
    vec2 uv = position.xz * causticsScale;
    vec2 movement = vec2(time * causticsSpeed * 0.3, time * causticsSpeed * 0.2);
    
    // Layer 1: Large-scale caustics
    float caustics1 = voronoiCaustics(uv * 0.5 + movement);
    
    // Layer 2: Medium-scale with different movement
    float caustics2 = voronoiCaustics(uv * 1.0 + movement * 1.3);
    
    // Layer 3: Small-scale detail
    float caustics3 = voronoiCaustics(uv * 2.0 + movement * 0.7);
    
    // Combine layers
    float caustics = caustics1 * 0.5 + caustics2 * 0.3 + caustics3 * 0.2;
    
    // Apply contrast and brightness
    caustics = smoothstep(0.1, 0.5, caustics);
    caustics = pow(caustics, 0.8);
    
    // Light direction modulation
    float NdotL = max(dot(vec3(0.0, 1.0, 0.0), -normalize(lightDir)), 0.0);
    
    // Add slight chromatic aberration to caustics
    vec3 causticsColor;
    causticsColor.r = caustics * 1.1;
    causticsColor.g = caustics;
    causticsColor.b = caustics * 0.9;
    
    return causticsColor * lightColor * causticsIntensity * depthFade * NdotL;
}

// Alternative: texture-based caustics sampling
vec3 sampleCausticsTexture(vec3 position) {
    float depth = waterSurfaceY - position.y;
    
    if(depth <= 0.0 || depth > maxDepth) {
        return vec3(0.0);
    }
    
    float depthFade = 1.0 - smoothstep(0.0, maxDepth, depth);
    
    vec2 uv = position.xz * causticsScale;
    vec2 movement = vec2(time * causticsSpeed);
    
    // Sample at two moving positions and blend
    vec3 caustics1 = texture(causticsTexture, uv + movement).rgb;
    vec3 caustics2 = texture(causticsTexture, uv * 1.2 - movement * 0.7).rgb;
    
    vec3 caustics = min(caustics1, caustics2); // Intersection pattern
    
    return caustics * lightColor * causticsIntensity * depthFade;
}

void main() {
    // Calculate procedural caustics
    vec3 caustics = calculateCaustics(worldPos);
    
    // Or use texture-based caustics (uncomment if texture available)
    // vec3 caustics = sampleCausticsTexture(worldPos);
    
    // Output as additive light contribution
    FragColor = vec4(caustics, 1.0);
}
