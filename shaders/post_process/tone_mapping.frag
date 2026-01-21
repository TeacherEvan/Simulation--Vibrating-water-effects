// Tone Mapping and Final Composite Shader for Panda3D
#version 430

uniform sampler2D hdrScene;
uniform sampler2D bloomTexture;
uniform sampler2D ssaoTexture;

uniform float exposure;
uniform float bloomIntensity;
uniform float gamma;
uniform bool enableBloom;
uniform bool enableSSAO;

in vec2 texcoord;
out vec4 fragColor;

// ACES Tone Mapping
vec3 acesToneMap(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Reinhard Tone Mapping (alternative)
vec3 reinhardToneMap(vec3 color) {
    return color / (color + vec3(1.0));
}

// Uncharted 2 Tone Mapping (alternative)
vec3 uncharted2ToneMapPartial(vec3 x) {
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 uncharted2ToneMap(vec3 color) {
    float exposureBias = 2.0;
    vec3 curr = uncharted2ToneMapPartial(color * exposureBias);
    vec3 W = vec3(11.2);
    vec3 whiteScale = vec3(1.0) / uncharted2ToneMapPartial(W);
    return curr * whiteScale;
}

void main() {
    // Sample HDR scene
    vec3 hdrColor = texture(hdrScene, texcoord).rgb;
    
    // Apply SSAO
    if (enableSSAO) {
        float ao = texture(ssaoTexture, texcoord).r;
        hdrColor *= ao;
    }
    
    // Apply bloom (additive)
    if (enableBloom) {
        vec3 bloomColor = texture(bloomTexture, texcoord).rgb;
        hdrColor += bloomColor * bloomIntensity;
    }
    
    // Exposure adjustment
    hdrColor *= exposure;
    
    // Tone mapping (ACES)
    vec3 mapped = acesToneMap(hdrColor);
    
    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / gamma));
    
    fragColor = vec4(mapped, 1.0);
}
