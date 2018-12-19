import OpenGL
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import *

from .constants import Shading
from .scene_object import PointCloudSceneObject, MeshSceneObject, InstancedSceneObject

class OpenGLObject(object):
    """Encapsulation of an OpenGL object (point cloud or mesh) on the GPU.

    Note
    ----
    Should only be created with an active OpenGL context.

    Attributes
    ----------
    obj : :obj:`SceneObject`
        The scene object being wrapped.
    is_instanced : bool
        True if the object is instanced, False if not.
    is_visible : bool
        True if the object is visible, False if not.
    is_transparent : bool
        True if the object is transparent, False if not.
    shading_mode : int
        Shading mode for object.
    """

    def __init__(self, obj):
        self.obj = obj
        self.shading_mode = obj.shading_mode

        self._buffers = []
        self._vaid = None

        # Create and Bind VAO
        self._vaid = glGenVertexArrays(1)
        glBindVertexArray(self._vaid)

        # Create vertex buffer, which contains all data needed at each vertex.
        # - Let:
        #     - VP : vertex position data (3-float)
        #     - VN : vertex normal data (3-float)
        #     - VC : vertex color data (4-float)
        #     - VT : vertex texture data (2-float)
        # - Then:
        #     - If a point cloud: [VP, VC]
        #     - Else:
        #         - If textured: [VP, VN, VT]
        #         - Elif vert colors or face colors: [VP, VN, VC]
        #         - Else: [VP, VN]
        vertexbuffer = glGenBuffers(1)
        self._buffers.append(vertex_buffer)
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer)
        attr_sizes = []
        if isinstance(obj, PointCloudSceneObject):
            # If a point cloud: [VP, VC]
            vc = obj.vertex_colors
            if vc.shape[1] == 3:
                vc = np.concatenate(vc, np.ones((vc.shape[0], vc.shape[1], 1)), axis=2)
            vertbuf = np.hstack((obj.vertices, vc)).astype(np.float32).flatten()
            attr_sizes = [3,4]
        elif isinstance(obj, MeshSceneObject):
            # Compute positions and normals
            if obj.material.smooth:
                vp = obj.vertices
                vn = obj.vertex_normals
            else:
                vp = obj.vertices[obj.faces].reshape(3*len(obj.faces), 3).astype
                vn = np.repeat(obj.face_normals, 3, axis=0)

            # If textured: [VP, VN, VT]
            if obj.shading_mode & Shading.TEX:
                if obj.material.smooth:
                    vt = obj.texture_coords
                else:
                    vt = obj.texture_coords[obj.faces].\
                            reshape(3*len(obj.faces), 2)
                vertbuf = np.hstack((vp, vn, vt)).astype(np.float32).flatten()
                attr_sizes = [3,3,2]
            # If colored: [VP, VN, VC]
            elif obj.shading_mode & Shading.COLORED:
                if obj.shading_mode & Shading.VERT_COLORS:
                    if obj.material.smooth:
                        vc = obj.vertex_colors.astype(np.float32)
                    else:
                        vc = obj.vertex_colors[obj.faces].\
                                reshape(3*len(obj.faces), obj.vertex_colors.shape[1])
                else:
                    if obj.material.smooth:
                        raise ValueError('Cannot have face colors for smooth mesh')
                    else:
                        vc = np.repeat(obj.face_colors, 3, axis=0)
                # Change vertex colors to 4-array
                if vc.shape[1] == 3:
                    vc = np.concatenate(vc, np.ones((vc.shape[0], vc.shape[1], 1)), axis=2)
                vertbuf = np.hstack((vp, vn, vc)).astype(np.float32).flatten()
                attr_sizes = [3,3,4]
            # If default:
            else:
                vertbuf = np.hstack((vp, vn)).astype(np.float32).flatten()
                attr_sizes = [3,3]

        # Create vertex buffer
        glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ*len(vertbuf), vertbuf, GL_STATIC_DRAW)
        total_sz = sum(attr_sizes)
        offset = 0
        for i, sz in enumerate(attr_sizes):
            glVertexAttribPointer(i, sz, GL_FLOAT, GL_FALSE, FLOAT_SZ*total_sz, ctypes.c_void_p(FLOAT_SZ*offset))
            glEnableVertexAttribArray(i)

        # If an instanced scene object, bind model matrix buffer
        if self.is_instanced:
            pose_data = obj.poses.flatten().astype(np.float32)
            modelbuffer = glGenBuffers(1)
            self._buffers.append(modelbuffer)
            glBindBuffer(GL_ARRAY_BUFFER, modelbuffer)
            glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ*len(pose_data), pose_data, GL_STATIC_DRAW)

            for i in range(0, 4):
                idx = i + len(attr_sizes)
                glEnableVertexAttribArray(idx)
                glVertexAttribPointer(idx, 4, GL_FLOAT, GL_FALSE, FLOAT_SZ*4, ctypes.c_void_p(4*FLOAT_SZ*i))
                glVertexAttribDivisor(idx, 1);

        # If a smooth mesh, bind element buffer
        if isinstance(obj, MeshSceneObject) and obj.material.smooth:
            elementbuffer = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, UINT_SZ*3*len(mesh.faces),
                         mesh.faces.flatten().astype(np.uint32), GL_STATIC_DRAW)
            self._buffers.append(elementbuffer)

        # Unbind VAO
        glBindVertexArray(0)

    def delete(self):
        # Delete buffer data
        glDeleteBuffers(len(self._buffers), self._buffers)

        # Delete vertex array
        glDeleteVertexArrays(1, [self._vaid])

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
