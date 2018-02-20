import ctypes
import numpy as np
import weakref

import pyglet
pyglet.options['shadow_window'] = False
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import *

from .constants import MAX_N_LIGHTS
from .light import AmbientLight, PointLight, DirectionalLight
from .shaders import vertex_shader, fragment_shader, depth_vertex_shader, depth_fragment_shader
from .scene_object import InstancedSceneObject

# Create static c_void_p objects to avoid leaking memory
C_VOID_PS = []
for i in range(5):
    C_VOID_PS.append(ctypes.c_void_p(4*4*i))

class OpenGLRenderer(object):
    """An OpenGL 3.0+ renderer, based on PyOpenGL.
    """

    def __init__(self, scene):
        """Initialize a renderer for a given scene.

        Parameters
        ----------
        scene : Scene
            A scene description.
        """
        self.scene = scene
        self._width = self.scene.camera.intrinsics.width
        self._height = self.scene.camera.intrinsics.height
        self._vaids = None
        self._colorbuf, self._depthbuf = None, None
        self._framebuf = None
        self._window = None

        # Initialize the OpenGL context with a 1x1 window and hide it immediately
        try:
            conf = pyglet.gl.Config(
                depth_size=24,
                double_buffer=True,
                major_version=3,
                minor_version=2
            )
            self._window = pyglet.window.Window(config=conf, visible=False, resizable=False, width=1, height=1)
        except:
            raise ValueError('Meshrender requires OpenGL 3+!')

        # Bind the frame buffer for offscreen rendering
        self._bind_frame_buffer()

        # Use the depth test functionality of OpenGL. Don't clip -- many normals may be backwards.
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

        # Load the meshes into VAO's
        self._vaids = self._load_meshes()

        # Load the shaders
        # Fix for pyopengl -- bind a framebuffer
        glBindVertexArray(self._vaids[0])
        self._full_shader = self._load_shaders(vertex_shader, fragment_shader)
        self._depth_shader = self._load_shaders(depth_vertex_shader, depth_fragment_shader)
        glBindVertexArray(0)

    @property
    def scene(self):
        return self._scene()

    @scene.setter
    def scene(self, s):
        self._scene = weakref.ref(s)

    def render(self, render_color=True, front_and_back=False):
        """Render raw images of the scene.

        Parameters
        ----------
        render_color : bool
            If True, both a color and a depth image are returned.
            If False, only a depth image is returned.

        front_and_back : bool
            If True, all normals are treated as facing the camera.

        Returns
        -------
        tuple of (h, w, 3) uint8, (h, w) float32
            A raw RGB color image with pixel values in [0, 255] and a depth image
            with true depths expressed as floats. If render_color was False,
            only the depth image is returned.

        Note
        -----
        This function can be called repeatedly, regardless of changes to the scene
        (i.e. moving SceneObjects, adding and removing lights, moving the camera).
        However, adding or removing objects causes a new OpenGL context to be created,
        so put all the objects in the scene before calling it.

        Note
        ----
        Values listed as 0.0 in the depth image are actually at infinity
        (i.e. no object present at that pixel).
        """
        # Reload the frame buffers if the width or height of the camera changed
        width = self.scene.camera.intrinsics.width
        height = self.scene.camera.intrinsics.height
        if width != self._width or height != self._height:
            self._width = width
            self._height = height
            self._bind_frame_buffer()

        if render_color:
            return self._color_and_depth(front_and_back)
        else:
            return self._depth()

    def close(self):
        """Destroy the OpenGL context attached to this renderer.

        Warning
        -------
        Once this has been called, the OpenGLRenderer object should be discarded.
        """
        if self._window is not None:
            OpenGL.contextdata.cleanupContext()
            self._window.close()
            self._window = None

    def _bind_frame_buffer(self):
        """Bind the frame buffer for offscreen rendering.
        """
        # Release the color and depth buffers if they exist:
        if self._framebuf is not None:
            glDeleteRenderbuffers(2, [self._colorbuf, self._depthbuf])
            glDeleteFramebuffers([self._framebuf])

        # Initialize the Framebuffer into which we will perform off-screen rendering
        self._colorbuf, self._depthbuf = glGenRenderbuffers(2)
        glBindRenderbuffer(GL_RENDERBUFFER, self._colorbuf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, self._width, self._height)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depthbuf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self._width, self._height)

        self._framebuf = glGenFramebuffers(1)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._framebuf)
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._colorbuf)
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depthbuf)

    def _load_shaders(self, vertex_shader, fragment_shader):
        """Load and compile shaders from strings.
        """
        shader = shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )

        return shader

    def _load_meshes(self):
        """Load the scene's meshes into vertex buffers.
        """
        VA_ids = glGenVertexArrays(len(self.scene.objects))

        if len(self.scene.objects) == 1:
            VA_ids = [VA_ids]

        null = C_VOID_PS[0]
        for VA_id, obj in zip(VA_ids, self.scene.objects.values()):
            mesh = obj.mesh
            material = obj.material

            glBindVertexArray(VA_id)

            if material.smooth:
                # If smooth is True, we use indexed element arrays and set only one normal per vertex.

                # Set up the vertex VBO
                vertexbuffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, null)
                glBufferData(GL_ARRAY_BUFFER,
                             4*3*len(mesh.vertices),
                             np.array(mesh.vertices.flatten(), dtype=np.float32),
                             GL_STATIC_DRAW)

                # Set up the normal VBO
                normalbuffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, normalbuffer)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, null)
                glBufferData(GL_ARRAY_BUFFER,
                             4*3*len(mesh.vertex_normals),
                             np.array(mesh.vertex_normals.flatten(), dtype=np.float32),
                             GL_STATIC_DRAW)

                # Set up the element index buffer
                elementbuffer = glGenBuffers(1)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                             4*3*len(mesh.faces),
                             np.array(mesh.faces.flatten(), dtype=np.int32),
                             GL_STATIC_DRAW)

            else:
                # If smooth is False, we treat each triangle independently
                # and set vertex normals to corresponding face normals.

                # Set up the vertices
                vertexbuffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, null)
                glBufferData(GL_ARRAY_BUFFER,
                             4*3*3*len(mesh.triangles),
                             np.array(mesh.triangles.flatten(), dtype=np.float32),
                             GL_STATIC_DRAW)

                # Set up the normals
                normalbuffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, normalbuffer)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, null)
                normals = np.array([[x,x,x] for x in mesh.face_normals], dtype=np.float32)
                normals = normals.flatten()
                glBufferData(GL_ARRAY_BUFFER,
                             4*len(normals),
                             normals,
                             GL_STATIC_DRAW)

            glVertexAttribDivisor(0, 0)
            glVertexAttribDivisor(1, 0)

            # Set up model matrix buffer
            modelbuf = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, modelbuf)
            for i in range(4):
                glEnableVertexAttribArray(2 + i)
                glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, 4*16, C_VOID_PS[i])
                glVertexAttribDivisor(2 + i, 1)

            if isinstance(obj, InstancedSceneObject):
                glBufferData(GL_ARRAY_BUFFER, 4*16*len(obj.poses), None, GL_STATIC_DRAW)
                data = np.concatenate([p.matrix.T for p in obj.poses], axis=0).flatten().astype(np.float32)
                glBufferSubData(GL_ARRAY_BUFFER, 0, 4*16*len(obj.poses), data)
            else:
                glBufferData(GL_ARRAY_BUFFER, 4*16, None, GL_STATIC_DRAW)
                glBufferSubData(GL_ARRAY_BUFFER, 0, 4*16, np.eye(4).flatten().astype(np.float32))


            # Unbind all buffers
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        return VA_ids

    def _depth(self):
        """Render a depth image of the scene.
        """
        camera = self.scene.camera
        width = camera.intrinsics.width
        height = camera.intrinsics.height

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._framebuf)
        glViewport(0, 0, width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._depth_shader)

        # Get Uniform Locations from Shader
        v_id = glGetUniformLocation(self._depth_shader, 'V')
        p_id = glGetUniformLocation(self._depth_shader, 'P')
        m_id = glGetUniformLocation(self._depth_shader, 'M')

        glUniformMatrix4fv(v_id, 1, GL_TRUE, camera.V)
        glUniformMatrix4fv(p_id, 1, GL_TRUE, camera.P)

        for vaid, obj in zip(self._vaids, self.scene.objects.values()):
            material = obj.material
            mesh = obj.mesh

            glUniformMatrix4fv(m_id, 1, GL_TRUE, obj.T_obj_world.matrix)

            glBindVertexArray(vaid)

            n_instances = 1
            if isinstance(obj, InstancedSceneObject):
                n_instances = len(obj.poses)

            if material.smooth:
                glDrawElementsInstanced(GL_TRIANGLES, 3*len(mesh.faces), GL_UNSIGNED_INT, C_VOID_PS[0], n_instances)
            else:
                glDrawArraysInstanced(GL_TRIANGLES, 0, 3*len(mesh.faces), n_instances)

            glBindVertexArray(0)

        glUseProgram(0)

        glFlush()

        # Extract the z buffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._framebuf)
        depth_buf = (GLfloat * (width * height))(0)
        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth_buf)

        # Re-format it into a numpy array
        depth_im = np.frombuffer(depth_buf, dtype=np.float32).reshape((height, width))
        depth_im = np.flip(depth_im, axis=0)
        inf_inds = (depth_im == 1.0)
        depth_im = 2.0 * depth_im - 1.0
        z_near, z_far = camera.z_near, camera.z_far
        depth_im = 2.0 * z_near * z_far / (z_far + z_near - depth_im * (z_far - z_near))
        depth_im[inf_inds] = 0.0

        return depth_im

    def _color_and_depth(self, front_and_back):
        """Render a color image and a depth image of the scene.
        """
        scene = self.scene
        camera = scene.camera
        width = camera.intrinsics.width
        height = camera.intrinsics.height

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._framebuf)
        glViewport(0, 0, width, height)

        glClearColor(.93, .93, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._full_shader)

        # Get Uniform Locations from Shader
        p_id = glGetUniformLocation(self._full_shader, 'P')
        v_id = glGetUniformLocation(self._full_shader, 'V')
        m_id = glGetUniformLocation(self._full_shader, 'M')
        matprop_id = glGetUniformLocation(self._full_shader, 'material_properties')
        object_color_id = glGetUniformLocation(self._full_shader, 'object_color')
        ambient_id = glGetUniformLocation(self._full_shader, 'ambient_light_info')
        directional_id = glGetUniformLocation(self._full_shader, "directional_light_info")
        n_directional_id = glGetUniformLocation(self._full_shader, "n_directional_lights")
        point_id = glGetUniformLocation(self._full_shader, "point_light_info")
        n_point_id = glGetUniformLocation(self._full_shader, "n_point_lights")
        front_and_back_id = glGetUniformLocation(self._full_shader, "front_and_back")

        # Bind bad normals id
        glUniform1i(front_and_back_id, int(front_and_back))

        # Bind view matrix
        glUniformMatrix4fv(v_id, 1, GL_TRUE, scene.camera.V)
        glUniformMatrix4fv(p_id, 1, GL_TRUE, scene.camera.P)

        # Bind ambient lighting
        glUniform4fv(ambient_id, 1, np.hstack((scene.ambient_light.color,
                                               scene.ambient_light.strength)))

        # Bind directional lighting
        glUniform1i(n_directional_id, len(scene.directional_lights))
        directional_info = np.zeros((2*MAX_N_LIGHTS, 4))
        for i, dlight in enumerate(scene.directional_lights):
            directional_info[2*i,:] = np.hstack((dlight.color, dlight.strength))
            directional_info[2*i+1,:] = np.hstack((dlight.direction, 0))
        glUniform4fv(directional_id, 2*MAX_N_LIGHTS, directional_info.flatten())

        # Bind point lighting
        glUniform1i(n_point_id, len(scene.point_lights))
        point_info = np.zeros((2*MAX_N_LIGHTS, 4))
        for i, plight in enumerate(scene.point_lights):
            point_info[2*i,:] = np.hstack((plight.color, plight.strength))
            point_info[2*i+1,:] = np.hstack((plight.location, 1))
        glUniform4fv(point_id, 2*MAX_N_LIGHTS, point_info.flatten())

        for vaid, obj in zip(self._vaids, scene.objects.values()):
            if not obj.enabled:
                continue

            mesh = obj.mesh
            material = obj.material

            glBindVertexArray(vaid)

            glUniformMatrix4fv(m_id, 1, GL_TRUE, obj.T_obj_world.matrix)
            glUniform3fv(object_color_id, 1, material.color)
            glUniform4fv(matprop_id, 1, np.array([material.k_a, material.k_d, material.k_s, material.alpha]))

            if material.wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            n_instances = 1
            if isinstance(obj, InstancedSceneObject):
                n_instances = len(obj.poses)

            if material.smooth:
                glDrawElementsInstanced(GL_TRIANGLES, 3*len(mesh.faces), GL_UNSIGNED_INT, C_VOID_PS[0], n_instances)
            else:
                glDrawArraysInstanced(GL_TRIANGLES, 0, 3*len(mesh.faces), n_instances)

            glBindVertexArray(0)

        glUseProgram(0)

        glFlush()

        # Extract the color and depth buffers
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._framebuf)
        color_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        depth_buf = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)

        # Re-format them into numpy arrays
        color_im = np.frombuffer(color_buf, dtype=np.uint8).reshape((height, width, 3))
        color_im = np.flip(color_im, axis=0)

        depth_im = np.frombuffer(depth_buf, dtype=np.float32).reshape((height, width))
        depth_im = np.flip(depth_im, axis=0)
        inf_inds = (depth_im == 1.0)
        depth_im = 2.0 * depth_im - 1.0
        z_near, z_far = camera.z_near, camera.z_far
        depth_im = 2.0 * z_near * z_far / (z_far + z_near - depth_im * (z_far - z_near))
        depth_im[inf_inds] = 0.0

        return color_im, depth_im

    def __del__(self):
        self.close()
