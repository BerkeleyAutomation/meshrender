#version 330 core

// Inputs
layout(location = 0) in vec3 position_a; // Model-frame vertex position
layout(location = 1) in vec3 normal_a;   // Model-frame vertex normal

// Output data
out VS_OUT {
    vec3 normal;
} vs_out;

// Uniform data
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Render loop
void main() {
    gl_Position = P * V * M * vec4(position_a, 1);
    position = vec3(M * vec4(position_a, 1.0));
    mat3 normal_matrix = mat3(transpose(inverse(V*M)));
    vs_out.normal = normalize(vec3(P * vec4(normal_matrix * normal_a, 0.0)));
}



