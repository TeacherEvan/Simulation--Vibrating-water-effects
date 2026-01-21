// Particle Fragment Shader
// Renders soft particles with various effects

#version 430 core

in vec2 fragTexCoord;
in vec4 fragColor;
in float fragLife;
in float fragSpeed;

out vec4 FragColor;

uniform sampler2D particleTexture;
uniform sampler2D depthTexture;
uniform sampler2D noiseTexture;

uniform vec2 screenSize;
uniform float nearPlane = 0.1;
uniform float farPlane = 1000.0;
uniform float softness = 1.0;  // Soft particle depth fade
uniform float time;

// Particle visual modes
uniform int visualMode = 0;  // 0=soft circle, 1=textured, 2=sparkle

// Linearize depth for soft particles
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
}

// Soft circle (procedural)
float softCircle(vec2 uv) {
    float dist = length(uv - 0.5) * 2.0;
    return 1.0 - smoothstep(0.5, 1.0, dist);
}

// Sparkle effect
float sparkle(vec2 uv, float t) {
    float dist = length(uv - 0.5) * 2.0;
    float ring = 1.0 - abs(dist - 0.5) * 4.0;
    ring = max(ring, 0.0);
    
    // Animated sparkle rays
    float angle = atan(uv.y - 0.5, uv.x - 0.5);
    float rays = pow(abs(sin(angle * 4.0 + t * 10.0)), 8.0);
    
    float core = 1.0 - smoothstep(0.0, 0.3, dist);
    
    return max(core, ring * rays) * (1.0 - smoothstep(0.8, 1.0, dist));
}

void main() {
    // Base alpha from particle shape
    float alpha;
    vec3 color = fragColor.rgb;
    
    if(visualMode == 0) {
        // Soft circle
        alpha = softCircle(fragTexCoord);
    } else if(visualMode == 1) {
        // Textured
        vec4 texColor = texture(particleTexture, fragTexCoord);
        alpha = texColor.a;
        color *= texColor.rgb;
    } else if(visualMode == 2) {
        // Sparkle
        alpha = sparkle(fragTexCoord, time);
    } else {
        alpha = softCircle(fragTexCoord);
    }
    
    // Apply particle color alpha
    alpha *= fragColor.a;
    
    // Fade based on life
    float lifeFade = smoothstep(0.0, 0.2, fragLife) * 
                     (1.0 - smoothstep(0.8, 1.0, 1.0 - fragLife));
    alpha *= lifeFade;
    
    // Soft particles - fade near geometry
    vec2 screenUV = gl_FragCoord.xy / screenSize;
    float sceneDepth = texture(depthTexture, screenUV).r;
    float sceneLinearDepth = linearizeDepth(sceneDepth);
    float particleLinearDepth = gl_FragCoord.z; // Linear depth of particle
    
    float depthDiff = sceneLinearDepth - particleLinearDepth;
    float softFade = smoothstep(0.0, softness, depthDiff);
    alpha *= softFade;
    
    // Speed-based effects
    if(fragSpeed > 2.0) {
        // Brighten fast particles
        color += vec3(0.1, 0.15, 0.2) * (fragSpeed - 2.0) * 0.1;
    }
    
    // Discard fully transparent fragments
    if(alpha < 0.01) {
        discard;
    }
    
    // Premultiplied alpha for additive blending
    FragColor = vec4(color * alpha, alpha);
}
