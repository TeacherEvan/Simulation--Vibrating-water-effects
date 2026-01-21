// Gaussian Blur Shader for Panda3D (Separable, Two-Pass)
#version 430

uniform sampler2D inputTexture;
uniform bool horizontal;

// 9-tap Gaussian weights (sigma ≈ 2.5)
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

in vec2 texcoord;
out vec4 fragColor;

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(inputTexture, 0));
    vec3 result = texture(inputTexture, texcoord).rgb * weights[0];
    
    vec2 offset;
    if (horizontal) {
        offset = vec2(texelSize.x, 0.0);
    } else {
        offset = vec2(0.0, texelSize.y);
    }
    
    for (int i = 1; i < 5; i++) {
        result += texture(inputTexture, texcoord + offset * float(i)).rgb * weights[i];
        result += texture(inputTexture, texcoord - offset * float(i)).rgb * weights[i];
    }
    
    fragColor = vec4(result, 1.0);
}
