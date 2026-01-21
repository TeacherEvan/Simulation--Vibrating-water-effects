// Glass Sphere Refraction Vertex Shader for Panda3D
#version 430

// Panda3D automatic uniforms
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
uniform vec3 wspos_camera;

// Custom uniforms for deformation
uniform sampler2D deformationMap;
uniform float deformationAmount;
uniform float time;
uniform float vibrationFrequency;

// Vertex inputs
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

// Outputs
out vec3 worldNormal;
out vec3 worldPos;
out vec3 viewDir;
out vec4 clipPos;
out vec2 texCoord;

void main() {
    vec3 position = p3d_Vertex.xyz;
    vec3 normal = p3d_Normal;
    
    // ============================================
    // SURFACE DEFORMATION (from simulation)
    // ============================================
    
    // Convert to spherical UV for deformation sampling
    vec2 sphereUV = vec2(
        atan(position.z, position.x) / (2.0 * 3.14159) + 0.5,
        acos(clamp(position.y / length(position), -1.0, 1.0)) / 3.14159
    );
    
    // Sample deformation from simulation
    float displacement = 0.0;
    if (deformationAmount > 0.0) {
        displacement = texture(deformationMap, sphereUV).r * deformationAmount;
    }
    
    // Add procedural vibration pattern
    float vibration = sin(vibrationFrequency * time + position.y * 10.0) * 0.005;
    displacement += vibration;
    
    // Apply displacement along normal
    position += normal * displacement;
    
    // ============================================
    // NORMAL RECALCULATION (approximate)
    // ============================================
    
    // For accurate normals, we'd need to compute finite differences
    // This is a simplified approach that works for small deformations
    float eps = 0.01;
    vec3 tangent = normalize(cross(normal, vec3(0.0, 1.0, 0.0)));
    if (length(tangent) < 0.1) {
        tangent = normalize(cross(normal, vec3(1.0, 0.0, 0.0)));
    }
    vec3 bitangent = cross(normal, tangent);
    
    // Perturb normal based on displacement gradient (simplified)
    vec2 gradUV = sphereUV + vec2(eps, 0.0);
    float dispRight = texture(deformationMap, gradUV).r * deformationAmount;
    gradUV = sphereUV + vec2(0.0, eps);
    float dispUp = texture(deformationMap, gradUV).r * deformationAmount;
    
    vec3 perturbedNormal = normalize(normal + 
        tangent * (displacement - dispRight) * 10.0 +
        bitangent * (displacement - dispUp) * 10.0);
    
    // ============================================
    // TRANSFORM TO WORLD SPACE
    // ============================================
    
    vec4 worldPosition = p3d_ModelMatrix * vec4(position, 1.0);
    worldPos = worldPosition.xyz;
    
    // Transform normal to world space
    worldNormal = normalize(p3d_NormalMatrix * perturbedNormal);
    
    // View direction (from surface to camera)
    viewDir = normalize(wspos_camera - worldPos);
    
    // Clip space position
    clipPos = p3d_ModelViewProjectionMatrix * vec4(position, 1.0);
    gl_Position = clipPos;
    
    // Texture coordinates
    texCoord = p3d_MultiTexCoord0;
}
