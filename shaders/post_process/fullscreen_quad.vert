// Full-Screen Quad Vertex Shader for Panda3D Post-Processing
#version 430

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

out vec2 texcoord;

void main() {
    gl_Position = p3d_Vertex;
    texcoord = p3d_MultiTexCoord0;
}
