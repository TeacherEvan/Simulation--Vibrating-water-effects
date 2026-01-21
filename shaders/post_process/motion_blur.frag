// Motion Blur Fragment Shader
// Per-pixel velocity-based motion blur

#version 430 core

in vec2 texCoords;

out vec4 FragColor;

uniform sampler2D colorTexture;
uniform sampler2D velocityTexture;  // RG = velocity in screen space
uniform sampler2D depthTexture;

uniform vec2 texelSize;
uniform float blurScale = 1.0;
uniform int numSamples = 8;
uniform float maxBlur = 20.0;  // Maximum blur in pixels

// Linearize depth
float linearizeDepth(float depth, float near, float far) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {
    // Read velocity (stored as screen-space motion per frame)
    vec2 velocity = texture(velocityTexture, texCoords).rg;
    
    // Scale velocity
    velocity *= blurScale;
    
    // Clamp velocity magnitude
    float speed = length(velocity);
    if(speed > maxBlur) {
        velocity = velocity / speed * maxBlur;
    }
    
    // Convert to texel offset
    vec2 texelVelocity = velocity * texelSize;
    
    // Skip if no motion
    if(length(texelVelocity) < 0.0001) {
        FragColor = texture(colorTexture, texCoords);
        return;
    }
    
    // Sample along velocity vector
    vec4 color = vec4(0.0);
    float totalWeight = 0.0;
    
    for(int i = 0; i < numSamples; i++) {
        float t = (float(i) / float(numSamples - 1)) - 0.5;
        vec2 sampleUV = texCoords + texelVelocity * t;
        
        // Clamp to screen bounds
        sampleUV = clamp(sampleUV, vec2(0.001), vec2(0.999));
        
        vec4 sampleColor = texture(colorTexture, sampleUV);
        
        // Weight by distance from center (optional, for softer blur)
        float weight = 1.0 - abs(t) * 0.5;
        
        color += sampleColor * weight;
        totalWeight += weight;
    }
    
    FragColor = color / totalWeight;
}
