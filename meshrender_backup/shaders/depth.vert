#version 330 core

// Inputs
layout(location = 0) in vec3 position_m;

// Output data
out vec4 color;

// Uniform data
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Render loop
void main() {
    gl_Position = P * V * M * vec4(position_m, 1);
    color = vec4(1.0);
}
