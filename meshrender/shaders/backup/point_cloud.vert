#version 330 core

// Inputs
layout(location = 0) in vec3 position_a;
layout(location = 1) in vec4 color_a;

// Output data
out vec4 color;

// Uniform data
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Render loop
void main() {
    gl_Position = P * V * M * vec4(position_a, 1);
    color = color_a;
}

