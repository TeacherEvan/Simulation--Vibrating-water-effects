// Temporal Anti-Aliasing Fragment Shader
// Reduces aliasing by blending with previous frame history

#version 430 core

in vec2 texCoords;

out vec4 FragColor;

uniform sampler2D currentFrame;
uniform sampler2D historyFrame;
uniform sampler2D velocityTexture;
uniform sampler2D depthTexture;

uniform vec2 texelSize;
uniform float blendFactor = 0.9;  // How much history to keep
uniform float velocityScale = 1.0;
uniform bool useYCoCg = true;  // Use YCoCg color space for better clamping

// Convert RGB to YCoCg
vec3 RGBToYCoCg(vec3 rgb) {
    float Y = dot(rgb, vec3(0.25, 0.5, 0.25));
    float Co = dot(rgb, vec3(0.5, 0.0, -0.5));
    float Cg = dot(rgb, vec3(-0.25, 0.5, -0.25));
    return vec3(Y, Co, Cg);
}

// Convert YCoCg to RGB
vec3 YCoCgToRGB(vec3 ycocg) {
    float Y = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    
    float r = Y + Co - Cg;
    float g = Y + Cg;
    float b = Y - Co - Cg;
    
    return vec3(r, g, b);
}

// Sample 3x3 neighborhood for variance clipping
void getNeighborhood(vec2 uv, out vec3 minColor, out vec3 maxColor, out vec3 avgColor) {
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    
    const int radius = 1;
    float weight = 0.0;
    
    for(int y = -radius; y <= radius; y++) {
        for(int x = -radius; x <= radius; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec3 color = texture(currentFrame, uv + offset).rgb;
            
            if(useYCoCg) {
                color = RGBToYCoCg(color);
            }
            
            m1 += color;
            m2 += color * color;
            weight += 1.0;
        }
    }
    
    // Calculate mean and variance
    avgColor = m1 / weight;
    vec3 variance = sqrt(max(m2 / weight - avgColor * avgColor, vec3(0.0)));
    
    // Variance clipping box
    float gamma = 1.0;  // Tighter = less ghosting, more aliasing
    minColor = avgColor - variance * gamma;
    maxColor = avgColor + variance * gamma;
}

// Clip color to AABB
vec3 clipAABB(vec3 color, vec3 minColor, vec3 maxColor) {
    vec3 center = (minColor + maxColor) * 0.5;
    vec3 extents = (maxColor - minColor) * 0.5;
    
    vec3 offset = color - center;
    vec3 v = offset / max(extents, vec3(0.0001));
    float maxVal = max(max(abs(v.x), abs(v.y)), abs(v.z));
    
    if(maxVal > 1.0) {
        return center + offset / maxVal;
    }
    
    return color;
}

void main() {
    // Sample velocity for reprojection
    vec2 velocity = texture(velocityTexture, texCoords).rg * velocityScale;
    
    // Reproject to previous frame position
    vec2 reprojectedUV = texCoords - velocity * texelSize;
    
    // Get current frame color
    vec3 currentColor = texture(currentFrame, texCoords).rgb;
    
    // Check if reprojected position is valid
    bool validHistory = reprojectedUV.x >= 0.0 && reprojectedUV.x <= 1.0 &&
                       reprojectedUV.y >= 0.0 && reprojectedUV.y <= 1.0;
    
    if(!validHistory) {
        FragColor = vec4(currentColor, 1.0);
        return;
    }
    
    // Sample history with bilinear filtering
    vec3 historyColor = texture(historyFrame, reprojectedUV).rgb;
    
    // Get neighborhood for variance clipping
    vec3 minColor, maxColor, avgColor;
    getNeighborhood(texCoords, minColor, maxColor, avgColor);
    
    // Convert colors for clamping
    vec3 currentYCoCg = useYCoCg ? RGBToYCoCg(currentColor) : currentColor;
    vec3 historyYCoCg = useYCoCg ? RGBToYCoCg(historyColor) : historyColor;
    
    // Clip history to neighborhood bounds
    vec3 clippedHistory = clipAABB(historyYCoCg, minColor, maxColor);
    
    // Convert back to RGB
    if(useYCoCg) {
        clippedHistory = YCoCgToRGB(clippedHistory);
    }
    
    // Calculate blend weight based on velocity
    float velocityWeight = 1.0 - smoothstep(0.0, 10.0, length(velocity));
    float adaptiveBlend = mix(0.5, blendFactor, velocityWeight);
    
    // Blend current and history
    vec3 finalColor = mix(currentColor, clippedHistory, adaptiveBlend);
    
    FragColor = vec4(finalColor, 1.0);
}
