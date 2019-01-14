#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in mat4 inst_m;

uniform mat4 light_matrix;
uniform mat4 M;

void main()
{
    gl_Position = light_matrix * M * inst_m * vec4(position, 1.0);
}
