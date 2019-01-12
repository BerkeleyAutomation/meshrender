#version 330 core
///////////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////////

struct SpotLight {
    vec3 color;
    float intensity;
    float range;
    vec3 position;
    vec3 direction;
    float light_angle_scale;
    float light_angle_offset;

    #ifdef SPOT_LIGHT_SHADOWS
    sampler2D shadow_map;
    #endif
};

struct DirectionalLight {
    vec3 color;
    float intensity;
    float range;
    vec3 direction;

    #ifdef DIRECTIONAL_LIGHT_SHADOWS
    sampler2D shadow_map;
    #endif
};

struct PointLight {
    vec3 color;
    float intensity;
    float range;
    vec3 position;

    #ifdef POINT_LIGHT_SHADOWS
    samplerCube shadow_map;
    #endif
};

struct Material {
    vec3 emissive_factor;

#ifdef METALLIC_MATERIAL
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
#endif

#ifdef GLOSSY_MATERIAL
    vec4 diffuse_factor;
    vec3 specular_factor;
    float glossiness_factor;
#endif

#ifdef NORMAL
    sampler2D normal_texture;
#endif
#ifdef OCCLUSION
    sampler2D occlusion_texture;
#endif
#ifdef EMISSIVE
    sampler2D emissive_texture;
#endif
#ifdef BASE_COLOR
    sampler2D base_color_texture;
#endif
#ifdef METALLIC_ROUGHNESS
    sampler2D metallic_roughness_texture;
#endif
#ifdef DIFFUSE
    sampler2D diffuse_texture;
#endif
#ifdef SPECULAR_GLOSSINESS
    sampler2D specular_glossiness;
#endif
};

struct PBRInfo {
    float nl;
    float nv;
    float nh;
    float lh;
    float vh;
    float roughness;
    float metallic;
    vec3 reflectance_0
    vec3 reflectance_90;
    vec3 diffuse_color;
    vec3 specular_color;
};

struct BRDFResult {
    vec3 F;
    float D;
    float G;
};

///////////////////////////////////////////////////////////////////////////////
// Uniforms
///////////////////////////////////////////////////////////////////////////////
uniform Material material;
uniform PointLight point_lights[MAX_POINT_LIGHTS];
uniform int n_point_lights;
uniform DirectionalLight directional_lights[MAX_POINT_LIGHTS];
uniform int n_directional_lights;
uniform SpotLight spot_lights[MAX_SPOT_LIGHTS];
uniform int n_spot_lights;

#ifdef USE_IBL
uniform samplerCube diffuse_env;
uniform samplerCube specular_env;
#endif

///////////////////////////////////////////////////////////////////////////////
// Inputs
///////////////////////////////////////////////////////////////////////////////
in vec3 frag_position;

#ifdef HAS_NORMALS
in vec3 frag_normal;
#ifdef NORMALS
in mat3 tbn;
#endif
#endif

#ifdef HAS_TEXCOORD_0
in vec2 uv;
#endif

#ifdef HAS_COLOR_0
out vec4 color_0;
#endif


///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////
const float PI = 3.141592653589793;
const float min_roughness = 0.04;

///////////////////////////////////////////////////////////////////////////////
// Utility Functions
///////////////////////////////////////////////////////////////////////////////
vec4 srgb_to_linear(vec4 srgb)
{
#ifndef SRGB_CORRECTED
    // Fast Approximation
    //vec3 linOut = pow(srgbIn.xyz,vec3(2.2));
    //
    vec3 b_less = step(vec3(0.04045),srgb.xyz);
    vec3 lin_out = mix( srgb.xyz/vec3(12.92), pow((srgb.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), b_less );
    return vec4(lin_out, srgb.w);
#else
    return srgb;
#endif
}

// Normal computation
vec3 get_normal()
{
#ifdef HAS_NORMALS
#ifdef NORMAL
    // TODO
    return frag_normal;
#else
    return frag_normal;
#endif
#else
    // default is pointing right at camera
    return normalize(vec3(V[3][0], V[3][1], V[3][2]) - frag_position);
#endif
}

// Fresnel
vec3 specular_reflection(PBRInfo info)
{
     vec3 res = info.reflectance_0 + (info.reflectance_90 - info.reflectance_0);
     res = res * pow(clamp(1.0 - info.vh, 0.0, 1.0), 5.0);
     return res;
}

// Smith
float geometric_occlusion(PBRInfo info)
{
    float nl = info.nl;
    float nv = info.nv;
    float r = info.roughness;
    float r2 = r * r;

    float a_l = 2.0 * nl / (nl + sqrt(r2 + (1.0 - r2) * (nl * nl)));
    float a_v = 2.0 * nv / (nv + sqrt(r2 + (1.0 - r2) * (nv * nv)));
    return a_l * a_v;
}

float microfacet_distribution(PBRInfo info)
{
    float r2 = info.roughness * info.roughness;
    float f = (info.nh * r2 - nh) * nh + 1.0;
    return r2 / (PI * f * f);
}

vec3 compute_brdf(vec3 n, vec3 v, vec3 l,
                  float roughness, float metalness,
                  float reflectance_0, float reflectance_90,
                  vec3 diffuse_color, vec3 specular_color,
                  vec3 radiance)
{
        vec3 h = normalize(l+v);
        float nl = clamp(dot(n, l), 0.001, 1.0);
        float nv = clamp(abs(dot(n, v)), 0.001, 1.0);
        float nh = clamp(dot(n, h), 0.0, 1.0);
        float lh = clamp(dot(l, h), 0.0, 1.0);
        float vh = clamp(dot(v, h), 0.0, 1.0);

        PBRInfo info = PBRInfo(
            nl, nv, nh, lh, vh, roughness, metalness,
            reflectance_0, reflectance_90,
            diffuse_color, specular_color
        );

        // Compute PBR terms
        vec3 F = specular_reflection(info);
        float G = geometric_occlusion(info);
        float D = microfacet_distribution(info);

        // Compute BRDF
        vec3 diffuse_contrib = (1.0 - F) * diffuse_color / PI;
        vec3 spec_contrib = F * G * D / (4.0 * nl * nv);

        vec3 color = nl * radiance * (diffuse_contrib + specular_contrib);
        return color
}

///////////////////////////////////////////////////////////////////////////////
// MAIN
///////////////////////////////////////////////////////////////////////////////
void main()
{

    vec3 color = vec3(0.0);
///////////////////////////////////////////////////////////////////////////////
// Handle Metallic Materials
///////////////////////////////////////////////////////////////////////////////
#ifdef METALLIC_MATERIAL

    // Compute metallic/roughness factors
    float roughness = material.roughness_factor;
    float metallic = material.metallic_factor;
#ifdef METALLIC_ROUGHNESS
    vec2 mr = texture2D(material.metallic_roughness_texture, uv);
    roughess = roughness * mr.r;
    metallic = metallic * mr.g;
#endif
    roughness = clamp(roughness, min_roughness, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    // In convention, material roughness is perceputal roughness ^ 2
    float alpha_roughness = roughness * roughness;

    // Compute albedo
    vec4 base_color = material.base_color_factor;
#ifdef BASE_COLOR
    base_color = base_color * srgb_to_linear(texture2D(material.base_color_texture, uv));
#endif

    // Compute specular and diffuse colors
    vec3 f0 = vec3(min_roughness);
    vec3 diffuse_color = base_color.rgb * (vec3(1.0) - f0);
    diffuse_color = diffuse_color * (1.0 - metallic);
    vec3 specular_color = mix(f0, base_color.rgb, metallic);

    // Compute reflectance
    // For typical incident reflectance range (between 4% to 100%) set the grazing reflectance to 100% for typical fresnel effect.
    // For very low reflectance range on highly diffuse objects (below 4%), incrementally reduce grazing reflecance to 0%.
    vec3 reflectance_0 = specular_color;
    vec3 reflectance_90 = vec3(1.0) * clamp(max(max(specular_color.r, specular_color.g), specular_color.b), 0.0, 1.0);

    // Compute normal
    vec3 n = get_normal();
    vec3 cam_pos = vec3(V[3][0], V[3][1], V[3][2]);

    // Loop over lights
    for (int i = 0; i < n_directional_lights; i++) {
        vec3 direction = directional_lights[i].direction;
        vec3 v = normalize(cam_pos - frag_position); // Vector towards camera
        vec3 l = normalize(-1.0 * direction);   // Vector towards light

        // Compute attenuation and radiance
        float attenuation = directional_lights[i].intensity;
        vec3 radiance = attenuation * directional_lights[i].color;

        // Compute outbound color
        vec3 res = compute_brdf(n, v, l, roughness, metallic,
                                reflectance_0, reflectance_90,
                                diffuse_color, specular_color,
                                radiance);
        color += res;
    }

    for (int i = 0; i < n_point_lights; i++) {
        vec3 position = point_lights[i].position;
        vec3 v = normalize(cam_pos - frag_position); // Vector towards camera
        vec3 l = normalize(position - frag_position); // Vector towards light

        // Compute attenuation and radiance
        float dist = length(position - frag_position);
        float attenuation = point_lights[i].intensity / (dist * dist);
        vec3 radiance = attenuation * directional_lights[i].color;

        // Compute outbound color
        vec3 res = compute_brdf(n, v, l, roughness, metallic,
                                reflectance_0, reflectance_90,
                                diffuse_color, specular_color,
                                radiance);
        color += res;
    }
    for (int i = 0; i < n_spot_lights; i++) {
        vec3 position = spot_lights[i].position;
        vec3 v = normalize(cam_pos - frag_position); // Vector towards camera
        vec3 l = normalize(position - frag_position); // Vector towards light

        // Compute attenuation and radiance
        vec3 direction = spot_lights[i].direction;
        float las = spot_lights[i].light_angle_scale;
        float lao = spot_lights[i].light_angle_offset;
        float dist = length(position - frag_position);
        float cd = clamp(dot(direction, -l), 0.0, 1.0);
        float attenuation = clamp(cd * las + lao, 0.0, 1.0);
        attenuation = attenuation * attenuation * spot_lights[i].intensity;
        attenuation = attenuation / (dist * dist);
        vec3 radiance = attenuation * point_lights[i].color;

        // Compute outbound color
        vec3 res = compute_brdf(n, v, l, roughness, metallic,
                                reflectance_0, reflectance_90,
                                diffuse_color, specular_color,
                                radiance);
        color += res;
    }

    // Calculate lighting from environment
#ifdef USE_IBL
    // TODO
#endif

    // Apply occlusion
#ifdef OCCLUSION
    float ao = texture2D(material.occlusion_texture, uv).r;
    color = color * ao;
#endif

    // Apply emissive map
    vec3 emissive = material.emissive_factor;
#ifdef EMISSIVE
    emissive *= srgb_to_linear(texture2D(material.emissive_texture, uv)).rgb;
#endif
    color += emissive * material.emissive_factor;

    gl_FragColor = vec4(pow(color, vec3(1.0/2.2)), base_color.a);

#endif

///////////////////////////////////////////////////////////////////////////////
// Handle Glossy Materials
///////////////////////////////////////////////////////////////////////////////

}
