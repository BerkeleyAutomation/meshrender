#version 330 core
// Attributes
attribute vec3 position;
#ifdef HAS_NORMALS
attribute vec3 normal;
#endif
#ifdef HAS_TANGENTS
attribute vec4 tangent;
#endif
#ifdef HAS_TEXCOORD_0
attribute vec2 texcoord_0;
#endif
#ifdef HAS_TEXCOORD_1
attribute vec2 texcoord_1;
#endif
#ifdef HAS_COLOR_0
attribute vec4 color_0;
#endif
attribute mat4 inst_m;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

out vec3 frag_position;

#ifdef HAS_NORMALS
out vec3 frag_normal;
#ifdef NORMALS
out mat3 tbn;
#endif
#endif

#ifdef HAS_TEXCOORD_0
out vec3 uv;
#endif

#ifdef HAS_COLOR_0
out vec4 color_multiplier;
#endif

void main()
{
    frag_position = vec3(M * inst_m * vec4(position_a, 1.0));

#ifdef HAS_NORMALS
    frag_normal = mat3(transpose(inverse(M))) * normal;
#ifdef HAS_TANGENTS
    // TODO
#endif
#endif

#ifdef HAS_TEXCOORD_0
    uv = texcoord_0;
#endif

#ifdef HAS_COLOR_0
    color_multiplier = color_0;
#endif

    gl_Position = P * V * M * inst_m * vec4(position, 1);
}
