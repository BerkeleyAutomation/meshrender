#version 330 core

// Constants
#define MAX_DIREC_LIGHTS 4
#define MAX_POINT_LIGHTS 4
#define MAX_SPOT_LIGHTS 4

// Structs
struct Material {
    vec4 diffuse;
    vec3 specular;
    float shininess;
    vec3 emission;
};

struct DirectionalLight {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 direction;
};

struct PointLight {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 position;
    float constant;
    float linear;
    float quadratic;
};

struct SpotLight {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 position;
    vec3 direction;
    float constant;
    float linear;
    float quadratic;
    float inner_angle;
    float outer_angle;
};

// Inputs
in vec3 position;
in vec3 normal;
in vec4 color;

// Outputs
out vec4 frag_color;

// Uniforms

uniform mat4 V;
uniform DirectionalLight directional_lights[MAX_DIREC_LIGHTS];
uniform PointLight point_lights[MAX_POINT_LIGHTS];
uniform SpotLight spot_lights[MAX_SPOT_LIGHTS];
uniform int n_direc_lights;
uniform int n_point_lights;
uniform int n_spot_lights;
uniform Material material;

// Function Prototypes
vec3 CalcDirLight(DirectionalLight light, vec3 normal, vec3 view_dir,
                  vec3 base_diffuse, vec3 base_specular);
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 position, vec3 view_dir,
                    vec3 base_diffuse, vec3 base_specular);
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 position, vec3 view_dir,
                   vec3 base_diffuse, vec3 base_specular);

void main() {
    vec3 norm = normalize(normal);
    vec3 view_dir = normalize(vec3(V[3][0], V[3][1], V[3][2]) - position);

    // Accumulate lighting
    vec3 result = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < n_direc_lights; i++) {
        result += CalcDirLight(directional_lights[i], norm, view_dir,
                               vec3(color), material.specular);
    }
    for (int i = 0; i < n_point_lights; i++) {
        result += CalcPointLight(point_lights[i], norm, position, view_dir,
                                 vec3(color), material.specular);
    }
    for (int i = 0; i < n_spot_lights; i++) {
        result += CalcSpotLight(spot_lights[i], norm, position, view_dir,
                                vec3(color), material.specular);
    }

    // Add emission
    result += material.emission;

    // Return fragment color
    frag_color = vec4(result, material.diffuse[3]);
}

// Calculates the color cast by a directional light
vec3 CalcDirLight(DirectionalLight light, vec3 normal,
                  vec3 view_dir, vec3 base_diffuse, vec3 base_specular)
{
    vec3 light_dir = normalize(-light.direction);
    vec3 halfway_dir = normalize(light_dir + view_dir);

    // Compute shading multipliers
    float diff = max(dot(normal, light_dir), 0.0);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);

    // Combine results
    vec3 ambient = light.ambient * base_diffuse;
    vec3 diffuse = light.diffuse * diff * base_diffuse;
    vec3 specular = light.specular * spec * base_specular;

    return ambient + diffuse + specular;
}

// Calculates the color cast by a point light
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 position,
                    vec3 view_dir, vec3 base_diffuse, vec3 base_specular)
{
    vec3 light_dir = normalize(light.position - position);
    vec3 halfway_dir = normalize(light_dir + view_dir);

    // Compute shading multipliers
    float diff = max(dot(normal, light_dir), 0.0);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
    float dist = length(light.position - position);
    float attenuation = 1.0 / (light.constant + light.linear * dist + light.quadratic * (dist * dist));

    // Combine results
    vec3 ambient = attenuation * light.ambient * base_diffuse;
    vec3 diffuse = attenuation * light.diffuse * diff * base_diffuse;
    vec3 specular = attenuation * light.specular * spec * base_specular;

    return ambient + diffuse + specular;
}

// Calculates the color cast by a point light
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 position,
                    vec3 view_dir, vec3 base_diffuse, vec3 base_specular)
{
    vec3 light_dir = normalize(light.position - position);
    vec3 halfway_dir = normalize(light_dir + view_dir);

    // Compute shading multipliers
    float diff = max(dot(normal, light_dir), 0.0);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
    float dist = length(light.position - position);
    float attenuation = 1.0 / (light.constant + light.linear * dist + light.quadratic * (dist * dist));
    float theta = dot(light_dir, normalize(-light.direction));
    float epsilon = light.inner_angle - light.outer_angle;
    float intensity = clamp((theta - light.outer_angle) / epsilon, 0.0, 1.0);

    // Combine results
    vec3 ambient = intensity * attenuation * light.ambient * base_diffuse;
    vec3 diffuse = intensity * attenuation * light.diffuse * diff * base_diffuse;
    vec3 specular = intensity * attenuation * light.specular * spec * base_specular;

    return ambient + diffuse + specular;
}

