// Bloom Brightness Extraction Shader for Panda3D
#version 430

uniform sampler2D hdrScene;
uniform float threshold;
uniform float softThreshold;

in vec2 texcoord;
out vec4 brightColor;

void main() {
    vec3 color = texture(hdrScene, texcoord).rgb;
    
    // Calculate luminance (ITU BT.709)
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    
    // Soft knee threshold for smooth transition
    float soft = brightness - threshold + softThreshold;
    soft = clamp(soft, 0.0, 2.0 * softThreshold);
    soft = soft * soft / (4.0 * softThreshold + 0.00001);
    
    float contribution = max(soft, brightness - threshold);
    contribution /= max(brightness, 0.00001);
    
    brightColor = vec4(color * contribution, 1.0);
}
