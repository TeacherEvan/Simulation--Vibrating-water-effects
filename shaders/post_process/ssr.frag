// Screen-Space Reflections Fragment Shader
// Raymarches through screen space to find reflections
// Optimized for real-time rendering at 60 FPS

#version 430 core

in vec2 texCoords;

out vec4 FragColor;

uniform sampler2D colorTexture;
uniform sampler2D depthTexture;
uniform sampler2D normalTexture;

uniform mat4 projMatrix;
uniform mat4 viewMatrix;
uniform mat4 invProjMatrix;
uniform mat4 invViewMatrix;

uniform vec2 screenSize;
uniform float maxDistance = 50.0;
uniform int maxSteps = 32;
uniform float thickness = 0.5;
uniform float stepStride = 1.0;
uniform float fadeEdge = 0.1;
uniform float roughnessThreshold = 0.5;

// Reconstruct view-space position from depth
vec3 reconstructViewPos(vec2 uv, float depth) {
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewPos = invProjMatrix * clipPos;
    return viewPos.xyz / viewPos.w;
}

// Project view-space position to screen space
vec3 projectToScreen(vec3 viewPos) {
    vec4 clipPos = projMatrix * vec4(viewPos, 1.0);
    vec3 ndcPos = clipPos.xyz / clipPos.w;
    return vec3(ndcPos.xy * 0.5 + 0.5, ndcPos.z * 0.5 + 0.5);
}

// Get view-space normal from normal texture
vec3 getViewNormal(vec2 uv) {
    vec3 worldNormal = texture(normalTexture, uv).xyz * 2.0 - 1.0;
    return normalize((viewMatrix * vec4(worldNormal, 0.0)).xyz);
}

// Binary search refinement for intersection
vec3 binarySearch(vec3 rayOrigin, vec3 rayDir, float rayLength) {
    float depth;
    vec3 projectedCoord;
    
    for(int i = 0; i < 8; i++) {
        projectedCoord = projectToScreen(rayOrigin + rayDir * rayLength);
        depth = texture(depthTexture, projectedCoord.xy).r;
        
        vec3 sampleViewPos = reconstructViewPos(projectedCoord.xy, depth);
        float deltaDepth = rayOrigin.z + rayDir.z * rayLength - sampleViewPos.z;
        
        if(deltaDepth > 0.0) {
            rayLength -= pow(0.5, float(i + 1));
        } else {
            rayLength += pow(0.5, float(i + 1));
        }
    }
    
    return projectedCoord;
}

// Main SSR raymarching
vec4 traceSSR(vec3 viewPos, vec3 viewNormal) {
    // Reflect the view direction around the normal
    vec3 viewDir = normalize(viewPos);
    vec3 reflectDir = reflect(viewDir, viewNormal);
    
    // Early exit if reflecting toward camera
    if(reflectDir.z > 0.0) {
        return vec4(0.0);
    }
    
    // Calculate step size based on distance
    float stepSize = stepStride;
    vec3 rayStep = reflectDir * stepSize;
    
    vec3 currentPos = viewPos + reflectDir * 0.1; // Offset to avoid self-intersection
    vec3 prevPos = currentPos;
    
    for(int i = 0; i < maxSteps; i++) {
        currentPos += rayStep;
        
        // Check if ray is too far
        float distanceTraveled = distance(viewPos, currentPos);
        if(distanceTraveled > maxDistance) {
            break;
        }
        
        // Project to screen space
        vec3 screenPos = projectToScreen(currentPos);
        
        // Check bounds
        if(screenPos.x < 0.0 || screenPos.x > 1.0 ||
           screenPos.y < 0.0 || screenPos.y > 1.0 ||
           screenPos.z < 0.0 || screenPos.z > 1.0) {
            break;
        }
        
        // Sample depth at this screen position
        float sampledDepth = texture(depthTexture, screenPos.xy).r;
        vec3 sampledViewPos = reconstructViewPos(screenPos.xy, sampledDepth);
        
        // Check for intersection
        float depthDiff = currentPos.z - sampledViewPos.z;
        
        if(depthDiff > 0.0 && depthDiff < thickness) {
            // Binary search for more precise hit
            vec3 hitCoord = binarySearch(prevPos, reflectDir, stepSize);
            
            // Fade at screen edges
            vec2 edgeFade = smoothstep(0.0, fadeEdge, hitCoord.xy) * 
                           (1.0 - smoothstep(1.0 - fadeEdge, 1.0, hitCoord.xy));
            float fade = edgeFade.x * edgeFade.y;
            
            // Fade based on distance traveled
            float distFade = 1.0 - clamp(distanceTraveled / maxDistance, 0.0, 1.0);
            
            // Fade based on view angle (grazing angles get more reflection)
            float angleFade = pow(1.0 - max(dot(-viewDir, viewNormal), 0.0), 2.0);
            
            vec3 reflectedColor = texture(colorTexture, hitCoord.xy).rgb;
            return vec4(reflectedColor, fade * distFade * angleFade);
        }
        
        prevPos = currentPos;
    }
    
    return vec4(0.0);
}

void main() {
    float depth = texture(depthTexture, texCoords).r;
    
    // Skip sky pixels
    if(depth >= 0.9999) {
        FragColor = vec4(0.0);
        return;
    }
    
    vec3 viewPos = reconstructViewPos(texCoords, depth);
    vec3 viewNormal = getViewNormal(texCoords);
    
    // Trace SSR
    vec4 ssrResult = traceSSR(viewPos, viewNormal);
    
    FragColor = ssrResult;
}
