import OpenGL
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import *

from .constants import Shading
from .scene_object import PointCloudSceneObject, MeshSceneObject

class OpenGLObject(object):

    def __init__(self, obj):
        self.obj = obj

        # Create VAO
        self.vaid = glGenVertexArrays(1)
        self.buffers = []

        # Bind VAO
        glBindVertexArray(self.vaid)

        # Create VBOs:
        #   - vertexbuffer: for vertex data (can include normal data,
        #                   texture data, and color data as needed)
        #   - elementbuffer: for face data (optional, only if smooth surface)
        #   - matrixbuffer: for instanced per-element model matrices

        # Create vertex data buffer
        vertex_data = []
        element_data = []
        if isinstance(obj, MeshSceneObject):
            # Using ElementBuffer and actual vertex normals
            if obj.material.smooth:
                vertex_data.append(obj.vertices.astype(np.float32))
                vertex_data.append(obj.vertex_normals.astype(np.float32))
                element_data.append(obj.faces)
            # Repeating vertices
            else:
                vertex_data.append(obj.vertices[obj.faces].reshape(3*len(obj.faces), 3).astype(np.float32))
                vertex_data.append(np.repeat(obj.face_normals, 3, axis=0).astype(np.float32))

            # Now, switch on shading modes to add more data
            # If doing textured shading, add texture coords
            if obj.shading_mode & Shading.TEX:
                vertex_data.append(obj.texture_coords.astype(np.float32))
            elif obj.shading_mode & Shading.VERT_COLORS:
                if obj.material.smooth:
                    vertex_data.append(obj.vertex_colors)
                else:
                    vertex_data.append(
                        obj.vertex_colors[obj.faces].\
                        reshape(3*obj.faces.shape[0], obj.vertex_colors.shape[1]).\
                        astype(np.float32)
                    )
            elif obj.shading_mode & Shading.FACE_COLORS:
                vertex_data.append(np.repeat(obj.face_colors, 3, axis=0).astype(np.float32))



        elif isinstance(obj, PointCloudSceneObject):
            vertex_data = np.hstack((obj.vertices, obj.vertex_colors))

        # Unbind VAO
        glBindVertexArray(0)


class GLManager(object):

    def __init__(self, use_framebuffer, width, height, background_color):
        self._use_framebuffer = use_framebuffer
        self._width = width
        self._height = height
        self._background_color = background_color

        self._init_basic_configuration()

        # Framebuffer with color and depth Renderbuffers
        self._framebuf = None
        self._colorbuf = None
        self._depthbuf = None

        if self.use_framebuffer:
            self._bind_framebuffer()

        # Map from scene objects to VAOs

    @property
    def use_framebuffer(self):
        return self._use_framebuffer

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, color):
        self._background_color = color
        glClearColor(color[0], color[1], color[2], 1.0)

    def resize(self, width, height):
        # TODO
        self.width = width
        self.height = height

        if self.use_framebuffer:
            self._bind_framebuffer()
        glViewport(0, 0, width, height)

    # Initialize basics

    def _init_basic_configuration(self):
        """Initialize simple, global OpenGL constants.
        """
        glClearColor(bg[0], bg[1], bg[2], 1.0)
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def _bind_framebuffer(self):
        """Bind the frame and render buffers for offscreen rendering.
        """
        # Remove old framebuffer
        self._delete_framebuffer()
        self._colorbuf, self._depthbuf = glGenRenderbuffers(2)
        glBindRenderbuffer(GL_RENDERBUFFER, self._colorbuf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, self.width, self.height)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depthbuf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)

        self._framebuf = glGenFramebuffers(1)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._framebuf)
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._colorbuf)
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depthbuf)

    def _delete_framebuffer(self):
        if self._framebuf is not None:
            glDeleteFramebuffers(1, [self._framebuf])
        if self._colorbuf is not None:
            glDeleteRenderbuffers(1, [self._colorbuf])
        if self._depthbuf is not None:
            glDeleteRenderbuffers(1, [self._depthbuf])

