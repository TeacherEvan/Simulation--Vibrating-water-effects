// Depth of Field Fragment Shader
// Implements bokeh-style blur with configurable parameters

#version 430 core

in vec2 texCoords;

out vec4 FragColor;

uniform sampler2D colorTexture;
uniform sampler2D depthTexture;

uniform float focusDistance = 10.0;
uniform float focalLength = 50.0;    // mm
uniform float aperture = 2.8;        // f-stop
uniform float filmHeight = 24.0;     // mm (35mm full frame)
uniform float nearPlane = 0.1;
uniform float farPlane = 1000.0;

uniform vec2 texelSize;
uniform int numSamples = 32;
uniform float maxBlur = 10.0;
uniform bool bokehEnabled = true;

const float PI = 3.14159265359;
const float GOLDEN_ANGLE = 2.39996323;

// Linearize depth from depth buffer
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
}

// Calculate Circle of Confusion (CoC) size
float calculateCoC(float depth) {
    // Thin lens equation for CoC diameter
    float S2 = depth;           // Subject distance
    float S1 = focusDistance;   // Focus distance
    float f = focalLength / 1000.0;  // Convert mm to meters
    float N = aperture;         // f-number
    
    // CoC = abs(A * f * (S1 - S2) / (S2 * (S1 - f)))
    // where A = f/N (aperture diameter)
    float A = f / N;
    float CoC = abs(A * f * (S1 - S2) / (S2 * (S1 - f)));
    
    // Convert to pixels (rough approximation)
    float pixelCoC = CoC * 1000.0 / filmHeight;
    
    return clamp(pixelCoC, 0.0, maxBlur);
}

// Bokeh kernel sampling positions (golden angle spiral)
vec2 getSamplePosition(int index, float coc) {
    float r = sqrt(float(index) + 0.5) / sqrt(float(numSamples));
    float theta = float(index) * GOLDEN_ANGLE;
    return vec2(cos(theta), sin(theta)) * r * coc;
}

// Hexagonal bokeh shape
float hexagonalWeight(vec2 offset, float coc) {
    if(!bokehEnabled) return 1.0;
    
    // Convert to polar coordinates
    float angle = atan(offset.y, offset.x);
    float dist = length(offset) / coc;
    
    // Create hexagonal shape
    float hexAngle = abs(mod(angle + PI/6.0, PI/3.0) - PI/6.0);
    float hexRadius = cos(PI/6.0) / cos(hexAngle);
    
    return smoothstep(hexRadius, hexRadius * 0.95, dist);
}

// Weighted bokeh blur
vec4 bokehBlur(vec2 uv, float coc) {
    vec4 color = vec4(0.0);
    float totalWeight = 0.0;
    
    for(int i = 0; i < numSamples; i++) {
        vec2 offset = getSamplePosition(i, coc) * texelSize;
        vec2 sampleUV = uv + offset;
        
        // Clamp to screen bounds
        sampleUV = clamp(sampleUV, vec2(0.001), vec2(0.999));
        
        vec4 sampleColor = texture(colorTexture, sampleUV);
        
        // Get depth at sample position
        float sampleDepth = linearizeDepth(texture(depthTexture, sampleUV).r);
        float sampleCoC = calculateCoC(sampleDepth);
        
        // Weight by bokeh shape and CoC overlap
        float weight = hexagonalWeight(offset / texelSize, coc);
        
        // Brighter samples get more weight for cat's eye effect
        weight *= 1.0 + pow(dot(sampleColor.rgb, vec3(0.299, 0.587, 0.114)), 2.0);
        
        color += sampleColor * weight;
        totalWeight += weight;
    }
    
    return color / max(totalWeight, 0.001);
}

// Fast separable approximation for performance
vec4 fastBlur(vec2 uv, float coc) {
    vec4 color = vec4(0.0);
    float totalWeight = 0.0;
    
    // Simple disc kernel
    int samples = min(numSamples, 16);
    
    for(int i = 0; i < samples; i++) {
        float angle = float(i) * 2.0 * PI / float(samples);
        float r = coc * texelSize.x;
        
        vec2 offset = vec2(cos(angle), sin(angle)) * r;
        vec4 sampleColor = texture(colorTexture, uv + offset);
        
        float weight = 1.0;
        color += sampleColor * weight;
        totalWeight += weight;
    }
    
    // Add center sample
    color += texture(colorTexture, uv) * 2.0;
    totalWeight += 2.0;
    
    return color / totalWeight;
}

void main() {
    // Get depth and calculate CoC
    float depth = texture(depthTexture, texCoords).r;
    float linearDepth = linearizeDepth(depth);
    float coc = calculateCoC(linearDepth);
    
    // Skip blur for in-focus regions
    if(coc < 0.5) {
        FragColor = texture(colorTexture, texCoords);
        return;
    }
    
    // Apply bokeh blur
    if(bokehEnabled && coc > 2.0) {
        FragColor = bokehBlur(texCoords, coc);
    } else {
        FragColor = fastBlur(texCoords, coc);
    }
}
