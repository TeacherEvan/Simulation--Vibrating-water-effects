// Water Surface Vertex Shader for Panda3D
// Handles wave displacement and coordinate generation
#version 430

// Panda3D automatic uniforms
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ViewMatrix;
uniform vec3 wspos_camera;  // Camera world position

// Custom uniforms
uniform float time;
uniform vec3 lightPos;

// Wave parameters (up to 4 Gerstner waves)
uniform vec4 waveA;  // xy = direction, z = steepness, w = wavelength
uniform vec4 waveB;
uniform vec4 waveC;
uniform vec4 waveD;

// Vertex inputs
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

// Outputs to fragment shader
out vec3 worldPos;
out vec4 clipSpace;
out vec2 texCoord;
out vec3 toCamera;
out vec3 fromLight;

const float PI = 3.14159265359;

// Gerstner wave function
vec3 gerstnerWave(vec4 wave, vec2 position, float t, inout vec3 tangent, inout vec3 binormal) {
    float steepness = wave.z;
    float wavelength = wave.w;
    
    float k = 2.0 * PI / wavelength;
    float c = sqrt(9.8 / k);
    vec2 d = normalize(wave.xy);
    float f = k * (dot(d, position) - c * t);
    float a = steepness / k;
    
    tangent += vec3(
        -d.x * d.x * steepness * sin(f),
        d.x * steepness * cos(f),
        -d.x * d.y * steepness * sin(f)
    );
    
    binormal += vec3(
        -d.x * d.y * steepness * sin(f),
        d.y * steepness * cos(f),
        -d.y * d.y * steepness * sin(f)
    );
    
    return vec3(
        d.x * a * cos(f),
        a * sin(f),
        d.y * a * cos(f)
    );
}

void main() {
    vec2 basePos = p3d_Vertex.xz;
    
    // Initialize tangent and binormal
    vec3 tangent = vec3(1.0, 0.0, 0.0);
    vec3 binormal = vec3(0.0, 0.0, 1.0);
    
    // Sum of all Gerstner waves
    vec3 displacement = vec3(0.0);
    
    if (waveA.w > 0.0) {
        displacement += gerstnerWave(waveA, basePos, time, tangent, binormal);
    }
    if (waveB.w > 0.0) {
        displacement += gerstnerWave(waveB, basePos, time, tangent, binormal);
    }
    if (waveC.w > 0.0) {
        displacement += gerstnerWave(waveC, basePos, time, tangent, binormal);
    }
    if (waveD.w > 0.0) {
        displacement += gerstnerWave(waveD, basePos, time, tangent, binormal);
    }
    
    // Apply displacement
    vec4 displacedVertex = p3d_Vertex;
    displacedVertex.x += displacement.x;
    displacedVertex.y += displacement.y;
    displacedVertex.z += displacement.z;
    
    // Calculate world position
    vec4 worldPosition = p3d_ModelMatrix * displacedVertex;
    worldPos = worldPosition.xyz;
    
    // Calculate clip-space position
    clipSpace = p3d_ModelViewProjectionMatrix * displacedVertex;
    gl_Position = clipSpace;
    
    // Texture coordinates (tiled)
    texCoord = p3d_MultiTexCoord0 * 4.0;
    
    // Vector to camera for Fresnel
    toCamera = wspos_camera - worldPos;
    
    // Vector from light
    fromLight = worldPos - lightPos;
}
