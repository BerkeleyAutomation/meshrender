DEFAULT_Z_NEAR = 0.05     # Near clipping plane, in meters
DEFAULT_Z_FAR = 100.0     # Far clipping plane, in meters
MAX_N_LIGHTS = 4  # Maximum number of lights allowed
OPEN_GL_MAJOR = 4 # Target OpenGL Major Version
OPEN_GL_MINOR = 1 # Target OpenGL Minor Version
FLOAT_SZ = 4      # Byte size of GL float32
UINT_SZ = 4       # Byte size of GL uint32

class GLTF(object):
    NEAREST = 9728
    LINEAR = 9729
    NEAREST_MIPMAP_NEAREST = 9984
    LINEAR_MIPMAP_NEAREST = 9985
    NEAREST_MIPMAP_LINEAR = 9986
    LINEAR_MIPMAP_LINEAR = 9987
    CLAMP_TO_EDGE = 33071
    MIRRORED_REPEAT = 33648
    REPEAT = 10497
    POINTS = 0
    LINES = 1
    LINE_LOOP = 2
    LINE_STRIP = 3
    TRIANGLES = 4
    TRIANGLE_STRIP = 5
    TRIANGLE_FAN = 6

class BufFlags(object):
    POSITION = 0
    NORMAL = 1
    TANGENT = 2
    TEXCOORD_0 = 4
    TEXCOORD_1 = 8
    COLOR_0 = 16
    JOINTS_0 = 32
    WEIGHTS_0 = 64

class TexFlags(object):
    NONE = 0
    NORMAL = 1
    OCCLUSION = 2
    EMISSIVE = 4
    BASE_COLOR = 8
    METALLIC_ROUGHNESS = 16
    DIFFUSE = 32
    SPECULAR_GLOSSINESS = 4

# Flags for render type
class RenderFlags(object):
    NONE = 0
    DEPTH_ONLY = 1
    SHADOWS = 2
    OFFSCREEN = 4
