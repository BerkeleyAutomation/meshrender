Z_NEAR = 0.05     # Near clipping plane, in meters
Z_FAR = 100.0     # Far clipping plane, in meters
MAX_N_LIGHTS = 10 # Maximum number of lights allowed
OPEN_GL_MAJOR = 4 # Target OpenGL Major Version
OPEN_GL_MINOR = 1 # Target OpenGL Minor Version
FLOAT_SZ = 4      # Byte size of GL float32
UINT_SZ = 4       # Byte size of GL uint32

# SHADING METHOD TYPES
class Shading(object):
    DEFUALT = 0
    POINT_CLOUD = 1
    FACE_COLORS = 2
    VERT_COLORS = 4
    TEX_DIFF = 8
    TEX_SPEC = 16
    TEX_NORM = 32
    TEX_EMIT = 64
    INSTANCED = 128
    VERT_NORMALS = 256

    # Other
    TEX = 8 | 16 | 32 | 64
    COLORED = 2 | 4
