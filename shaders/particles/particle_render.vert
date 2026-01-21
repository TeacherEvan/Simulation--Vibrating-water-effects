// Instanced Particle Vertex Shader
// Renders GPU particles as camera-facing billboards

#version 430 core

// Per-vertex attributes (quad)
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;

// Per-instance attributes (from particle buffer)
layout(location = 2) in vec4 instancePosition;  // xyz + life
layout(location = 3) in vec4 instanceVelocity;  // xyz + size
layout(location = 4) in vec4 instanceColor;     // rgba

out vec2 fragTexCoord;
out vec4 fragColor;
out float fragLife;
out float fragSpeed;

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ViewMatrix;
uniform vec3 cameraPos;
uniform vec3 cameraUp;
uniform vec3 cameraRight;

void main() {
    // Skip dead particles
    if(instancePosition.w <= 0.0) {
        gl_Position = vec4(0.0, 0.0, -1000.0, 1.0);
        return;
    }
    
    // Get particle properties
    vec3 particlePos = instancePosition.xyz;
    float life = instancePosition.w;
    float size = instanceVelocity.w;
    vec4 color = instanceColor;
    
    // Calculate billboard orientation
    vec3 toCamera = normalize(cameraPos - particlePos);
    vec3 right = cameraRight;
    vec3 up = cameraUp;
    
    // Apply billboard offset
    vec3 vertexPos = particlePos + 
                     right * position.x * size + 
                     up * position.y * size;
    
    // Calculate speed for stretching (optional)
    float speed = length(instanceVelocity.xyz);
    
    // Apply velocity stretching for fast particles
    if(speed > 1.0) {
        vec3 velDir = normalize(instanceVelocity.xyz);
        float stretchFactor = min(speed * 0.1, 2.0);
        
        // Stretch along velocity direction
        float alignment = dot(normalize(position.xy), vec2(0.0, 1.0));
        vertexPos += velDir * alignment * size * stretchFactor;
    }
    
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(vertexPos, 1.0);
    
    fragTexCoord = texcoord;
    fragColor = color;
    fragLife = life;
    fragSpeed = speed;
}
