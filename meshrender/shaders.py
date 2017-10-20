'''Shaders for pairing with the renderer.
'''

depth_vertex_shader = '''#version 330 core

// Input vertex data
layout(location = 0) in vec3 vertex_position_m;

// Output data
out vec4 color;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main(){
    gl_Position =  MVP * vec4(vertex_position_m, 1);

    color = vec4(1.0);
}
'''

depth_fragment_shader = '''#version 330 core

// Interpolated values from the vertex shaders
in vec4 color;

out vec4 frag_color;

void main(){
    frag_color = color;
}
'''

vertex_shader = '''#version 330 core

// Input vertex data
layout(location = 0) in vec3 vertex_position_m;
layout(location = 1) in vec3 vertex_normal_m;

// Output data
out vec4 color;
out vec4 position;
out vec3 normal;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;
uniform mat4 MV;
uniform vec3 object_color;

void main(){
    gl_Position =  MVP * vec4(vertex_position_m, 1);

    color = vec4(object_color, 1.0);
    position = MV * vec4(vertex_position_m, 1);
    normal = normalize(MV * vec4(vertex_normal_m, 0)).xyz;
}
'''

fragment_shader = '''#version 330 core

const int MAX_N_LIGHTS = 10;

// Interpolated values from the vertex shaders
in vec4 color;
in vec4 position;
in vec3 normal;

out vec4 frag_color;

uniform vec4 material_properties;
uniform vec4 ambient_light_info; 
uniform int n_point_lights;
uniform int n_directional_lights;
uniform vec4 point_light_info[2*MAX_N_LIGHTS];
uniform vec4 directional_light_info[2*MAX_N_LIGHTS];
uniform mat4 V;

void main(){

    // Extract material properties
    float k_a = material_properties[0];   // Ambient reflection constant
    float k_d = material_properties[1];   // Diffuse reflection constant
    float k_s = material_properties[2];   // Specular reflection constant
    float alpha = material_properties[3]; // Shininess

    // Compute Lighting Intensities
    float ambient_strength = ambient_light_info[3];
    vec3 ambient_color = vec3(ambient_light_info);

    vec3 i_ambient = ambient_strength * ambient_color;
    vec3 i_diffuse = vec3(0.0);
    vec3 i_specular = vec3(0.0);

    vec3 n = normalize(normal);
    vec3 e = normalize(-vec3(position));

    // Directional lights
    for (int i = 0; i < n_directional_lights; i++) {
        vec3 light_color = vec3(directional_light_info[2*i]);
        float light_strength = directional_light_info[2*i][3];
        light_color = light_color * light_strength;

        vec3 l = normalize(-vec3(V * directional_light_info[2*i+1]));

        vec3 r = reflect(-l, n);
        float diffuse = clamp(dot(n, l), 0, 1);
        float specular = clamp(dot(e, r), 0, 1);
        if (specular > 0.0) {
            specular = pow(specular, alpha);
        }

        i_diffuse += light_color * diffuse;
        i_specular += light_color * specular;
    }

    // Point lights
    for (int i = 0; i < n_point_lights; i++) {
        vec3 light_color = vec3(point_light_info[2*i]);
        float light_strength = point_light_info[2*i][3];
        light_color = light_color * light_strength;

        vec3 l = vec3(V * point_light_info[2*i+1]) - vec3(position);
        float dist = length(l);
        l = l / dist;

        light_color *= 1.0 / (dist * dist);

        vec3 r = reflect(-l, n);
        float diffuse = clamp(dot(n, l), 0, 1);
        float specular = clamp(dot(e, r), 0, 1);
        if (specular > 0.0) {
            specular = pow(specular, alpha);
        }

        i_diffuse += light_color * diffuse;
        i_specular += light_color * specular;
    }

    // Compute final pixel color
    frag_color = vec4((i_ambient * vec3(color) * k_a) + // Ambient
                      (i_diffuse * vec3(color) * k_d) + // Diffuse
                      (i_specular * k_s), 1.0);         // Specular (unweighted by shape color)
}
'''
