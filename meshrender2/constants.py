Z_NEAR = 0.05     # Near clipping plane, in meters
Z_FAR = 100.0     # Far clipping plane, in meters
MAX_N_LIGHTS = 4  # Maximum number of lights allowed
OPEN_GL_MAJOR = 4 # Target OpenGL Major Version
OPEN_GL_MINOR = 1 # Target OpenGL Minor Version
FLOAT_SZ = 4      # Byte size of GL float32
UINT_SZ = 4       # Byte size of GL uint32


# Type of vertex data available (in order of appearance)
class VertexBufferFlags(object):
    POSITION = 0    # Vertex position data available as 3-float
    NORMAL = 1      # Vertex normal data available as 3-float
    COLOR = 2       # Vertex color data available as 4-float
    TEXTURE = 4     # Vertex texture coordinates available as 2-float
    TANGENT = 8     # Tangent directions are available as 3-float
    BITANGENT = 16  # Bitangent directions are available as 3-float

# Type of rendering to be done
class VertexArrayFlags(object):
    POINTS = 0      # Render as points (mutually-exclusive with TRIANGLES)
    TRIANGLES = 1   # Render as triangles (mutually-exclusive with POINTS)
    INSTANCED = 2   # Instanced render with pose buffer
    ELEMENTS = 4    # Render as triangles with element buffer

# Type of textures available to object (in order of appearance)
class TextureFlags(object):
    NONE = 0        # No texture data available
    DIFFUSE = 1     # Diffuse texture map available (4-float)
    SPECULAR = 2    # Specular texture map available (1-float)
    EMISSION = 4    # Emission texture map available (3-float)
    NORMAL = 8      # Normal bump map available (3-float)
    HEIGHT = 16     # Height/displacement map available (1-float)

# Flags for render type
class RenderFlags(object):
    NONE = 0
    DEPTH_ONLY = 1
    SHADOWS = 2
    OFFSCREEN = 4
