// Chromatic Aberration Fragment Shader
// Separates RGB channels based on distance from center

#version 430 core

in vec2 texCoords;

out vec4 FragColor;

uniform sampler2D colorTexture;
uniform float strength = 0.005;
uniform vec2 direction = vec2(1.0, 0.0);  // Radial if (0,0)
uniform bool radial = true;
uniform float falloff = 2.0;  // Power curve for radial falloff

void main() {
    vec2 center = vec2(0.5);
    vec2 toCenter = texCoords - center;
    float dist = length(toCenter);
    
    vec2 offset;
    if(radial) {
        // Radial aberration (stronger at edges)
        offset = normalize(toCenter) * strength * pow(dist * 2.0, falloff);
    } else {
        // Directional aberration
        offset = direction * strength;
    }
    
    // Sample each channel at slightly different positions
    float r = texture(colorTexture, texCoords + offset).r;
    float g = texture(colorTexture, texCoords).g;
    float b = texture(colorTexture, texCoords - offset).b;
    
    FragColor = vec4(r, g, b, 1.0);
}
