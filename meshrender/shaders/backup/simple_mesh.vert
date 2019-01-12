#version 330 core

// Structs
struct Material {
    vec4 diffuse;
    vec3 specular;
    float shininess;
    vec3 emission;
};

// Inputs
layout(location = 0) in vec3 position_a; // Model-frame vertex position
layout(location = 1) in vec3 normal_a;   // Model-frame vertex normal
layout(location = 2) in mat4 inst_m_a;   // Instance pose matrix

// Output data
out vec3 position;  // World-frame vertex position
out vec3 normal;    // World-frame vertex normal
out vec4 color;     // RGBA color for vertex

// Uniform data
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform Material material;

// Render loop
void main() {
    mat4 M_full = M * inst_m_a;
    gl_Position = P * V * M_full * vec4(position_a, 1);

    position = vec3(M_full * vec4(position_a, 1.0));
    normal = mat3(transpose(inverse(M_full))) * normal_a;
    color = material.diffuse;
}

