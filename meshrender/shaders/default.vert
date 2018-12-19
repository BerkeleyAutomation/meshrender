#version 330 core

// Inputs
layout(location = 0) in vec3 position_a; // Model-frame vertex position
layout(location = 1) in vec3 normal_a;   // Model-frame vertex normal

// Output data
out vec3 position;  // World-frame vertex position
out vec3 normal;    // World-frame vertex normal

// Uniform data
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Render loop
void main() {
    gl_Position = P * V * M * vec4(position_a, 1);

    position = vec3(M * vec4(position_a, 1.0));
    normal = mat3(transpose(inverse(M))) * normal_a;
}


