// God Rays / Volumetric Light Scattering Shader
// Based on screen-space radial blur from light source

#version 430 core

in vec2 texCoords;

out vec4 FragColor;

uniform sampler2D colorTexture;
uniform sampler2D occlusionTexture;  // Scene rendered with objects as black

uniform vec2 lightScreenPos;  // Light position in screen space [0,1]
uniform float exposure = 0.0034;
uniform float decay = 0.97;
uniform float density = 0.84;
uniform float weight = 5.65;
uniform int numSamples = 64;

uniform vec3 lightColor = vec3(1.0, 0.95, 0.8);
uniform float lightIntensity = 1.0;

void main() {
    // Calculate ray from current pixel to light
    vec2 deltaTexCoord = (texCoords - lightScreenPos);
    deltaTexCoord *= 1.0 / float(numSamples) * density;
    
    // Initial values
    vec2 uv = texCoords;
    float illuminationDecay = 1.0;
    vec3 godRays = vec3(0.0);
    
    // Raymarch toward light source
    for(int i = 0; i < numSamples; i++) {
        uv -= deltaTexCoord;
        
        // Clamp to screen bounds
        if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }
        
        // Sample occlusion (bright where light shines through)
        vec3 sampleColor = texture(occlusionTexture, uv).rgb;
        
        // Accumulate with decay
        sampleColor *= illuminationDecay * weight;
        godRays += sampleColor;
        
        // Apply decay
        illuminationDecay *= decay;
    }
    
    // Apply exposure and color
    godRays *= exposure * lightColor * lightIntensity;
    
    // Add to original scene
    vec4 sceneColor = texture(colorTexture, texCoords);
    FragColor = sceneColor + vec4(godRays, 0.0);
}
