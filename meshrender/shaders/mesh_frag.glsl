#version 330 core
///////////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////////
#define MAX_SPOT_LIGHTS 4
#define MAX_DIRECTIONAL_LIGHTS 4
#define MAX_POINT_LIGHTS 4

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

#ifdef USE_METALLIC_MATERIAL
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
#endif

#ifdef USE_GLOSSY_MATERIAL
    vec4 diffuse_factor;
    vec3 specular_factor;
    float glossiness_factor;
#endif

#ifdef HAS_NORMAL_TEX
    sampler2D normal_texture;
#endif
#ifdef HAS_OCCLUSION_TEX
    sampler2D occlusion_texture;
#endif
#ifdef HAS_EMISSIVE_TEX
    sampler2D emissive_texture;
#endif
#ifdef HAS_BASE_COLOR_TEX
    sampler2D base_color_texture;
#endif
#ifdef HAS_METALLIC_ROUGHNESS_TEX
    sampler2D metallic_roughness_texture;
#endif
#ifdef HAS_DIFFUSE_TEX
    sampler2D diffuse_texture;
#endif
#ifdef HAS_SPECULAR_GLOSSINESS_TEX
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
    vec3 f0;
    vec3 c_diff;
};

///////////////////////////////////////////////////////////////////////////////
// Uniforms
///////////////////////////////////////////////////////////////////////////////
uniform Material material;
uniform PointLight point_lights[MAX_POINT_LIGHTS];
uniform int n_point_lights;
uniform DirectionalLight directional_lights[MAX_DIRECTIONAL_LIGHTS];
uniform int n_directional_lights;
uniform SpotLight spot_lights[MAX_SPOT_LIGHTS];
uniform int n_spot_lights;
uniform vec3 cam_pos;

#ifdef USE_IBL
uniform samplerCube diffuse_env;
uniform samplerCube specular_env;
#endif

///////////////////////////////////////////////////////////////////////////////
// Inputs
///////////////////////////////////////////////////////////////////////////////

in vec3 frag_position;
#ifdef NORMAL_LOC
in vec3 frag_normal;
#endif
#ifdef HAS_NORMAL_TEX
#ifdef TANGENT_LOC
#ifdef NORMAL_LOC
in mat3 tbn;
#endif
#endif
#endif
#ifdef TEXCOORD_0_LOC
in vec2 uv_0;
#endif
#ifdef TEXCOORD_1_LOC
in vec2 uv_1;
#endif
#ifdef COLOR_0_LOC
in vec4 color_multiplier;
#endif

///////////////////////////////////////////////////////////////////////////////
// OUTPUTS
///////////////////////////////////////////////////////////////////////////////

out vec4 frag_color;

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
#ifdef HAS_NORMAL_TEX
#ifndef HAS_TANGENTS
    vec3 pos_dx = dFdx(frag_position);
    vec3 pos_dy = dFdy(frag_position);
    vec3 tex_dx = dFdx(vec3(uv_0, 0.0));
    vec3 tex_dy = dFdy(vec3(uv_0, 0.0));
    vec3 t = (tex_dy.t * pos_dx - tex_dx.t * pos_dy) / (tex_dx.s * tex_dy.t - tex_dy.s * tex_dx.t);

#ifdef NORMAL_LOC
    vec3 ng = normalize(frag_normal);
#else
    vec3 = cross(pos_dx, pos_dy);
#endif

    t = normalize(t - ng * dot(ng, t));
    vec3 b = normalize(cross(ng, t));
    mat3 tbn_n = mat3(t, b, ng);
#else
    mat3 tbn_n = tbn;
#endif
    vec3 n = texture2D(material.normal_texture, uv_0).rgb;
    n = normalize(tbn_n * ((2.0 * n - 1.0) * vec3(1.0, 1.0, 1.0)));
    return n; // TODO NORMAL MAPPING
#else
#ifdef NORMAL_LOC
    return frag_normal;
#else
    return normalize(cam_pos - frag_position);
#endif
#endif
}

// Fresnel
vec3 specular_reflection(PBRInfo info)
{
     vec3 res = info.f0 + (1.0 - info.f0) * pow(clamp(1.0 - info.vh, 0.0, 1.0), 5.0);
     return res;
}

// Smith
float geometric_occlusion(PBRInfo info)
{
    float r = info.roughness + 1.0;
    float k = r * r  / 8.0;
    float g1 = info.nv / (info.nv * (1.0 - k) + k);
    float g2 = info.nl / (info.nl * (1.0 - k) + k);
    //float k = info.roughness * sqrt(2.0 / PI);
    //float g1 = info.lh / (info.lh * (1.0 - k) + k);
    //float g2 = info.nh / (info.nh * (1.0 - k) + k);
    return g1 * g2;
}

float microfacet_distribution(PBRInfo info)
{
    float a = info.roughness * info.roughness;
    float a2 = a * a;
    float nh2 = info.nh * info.nh;

    float denom = (nh2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

vec3 compute_brdf(vec3 n, vec3 v, vec3 l,
                  float roughness, float metalness,
                  vec3 f0, vec3 c_diff, vec3 albedo,
                  vec3 radiance)
{
        vec3 h = normalize(l+v);
        float nl = clamp(dot(n, l), 0.001, 1.0);
        float nv = clamp(abs(dot(n, v)), 0.001, 1.0);
        float nh = clamp(dot(n, h), 0.0, 1.0);
        float lh = clamp(dot(l, h), 0.0, 1.0);
        float vh = clamp(dot(v, h), 0.0, 1.0);

        PBRInfo info = PBRInfo(nl, nv, nh, lh, vh, roughness, metalness, f0, c_diff);

        // Compute PBR terms
        vec3 F = specular_reflection(info);
        float G = geometric_occlusion(info);
        float D = microfacet_distribution(info);

        // Compute BRDF
        vec3 diffuse_contrib = (1.0 - F) * c_diff / PI;
        vec3 spec_contrib = F * G * D / (4.0 * nl * nv + 0.001);

        vec3 color = nl * radiance * (diffuse_contrib + spec_contrib);
        return color;
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
#ifdef USE_METALLIC_MATERIAL

    // Compute metallic/roughness factors
    float roughness = material.roughness_factor;
    float metallic = material.metallic_factor;
#ifdef HAS_METALLIC_ROUGHNESS_TEX
    vec2 mr = texture2D(material.metallic_roughness_texture, uv_0).rg;
    roughness = roughness * mr.r;
    metallic = metallic * mr.g;
#endif
    roughness = clamp(roughness, min_roughness, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    // In convention, material roughness is perceputal roughness ^ 2
    float alpha_roughness = roughness * roughness;

    // Compute albedo
    vec4 base_color = material.base_color_factor;
#ifdef HAS_BASE_COLOR_TEX
    base_color = base_color * srgb_to_linear(texture2D(material.base_color_texture, uv_0));
#endif

    // Compute specular and diffuse colors
    vec3 dialectric_spec = vec3(min_roughness);
    vec3 c_diff = mix(vec3(0.0), base_color.rgb * (1 - min_roughness), 1.0 - metallic);
    vec3 f0 = mix(dialectric_spec, base_color.rgb, metallic);

    // Compute reflectance
    // For typical incident reflectance range (between 4% to 100%) set the grazing reflectance to 100% for typical fresnel effect.
    // For very low reflectance range on highly diffuse objects (below 4%), incrementally reduce grazing reflecance to 0%.

    // Compute normal
    vec3 n = normalize(get_normal());

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
                                f0, c_diff, base_color.rgb, radiance);
        color += res;
        //if (dot(n, l) < 0.8) {
        //    color += vec3(0.0);
        //} else {
        //    color += vec3(base_color);
        //}
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
                                f0, c_diff, base_color.rgb, radiance);
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
                                f0, c_diff, base_color.rgb, radiance);
        color += res;
    }

    // Calculate lighting from environment
#ifdef USE_IBL
    // TODO
#endif

    // Apply occlusion
#ifdef HAS_OCCLUSION_TEX
    float ao = texture2D(material.occlusion_texture, uv_0).r;
    color = color * ao;
#endif

    // Apply emissive map
    vec3 emissive = material.emissive_factor;
#ifdef HAS_EMISSIVE_TEX
    emissive *= srgb_to_linear(texture2D(material.emissive_texture, uv_0)).rgb;
#endif
    color += emissive * material.emissive_factor;

#ifdef COLOR_0_LOC
    color *= color_multiplier;
#endif

    frag_color = clamp(vec4(pow(color, vec3(1.0/2.2)), base_color.a), 0.0, 1.0);

#else
    // TODO GLOSSY MATERIAL BRDF
#endif

///////////////////////////////////////////////////////////////////////////////
// Handle Glossy Materials
///////////////////////////////////////////////////////////////////////////////

}
