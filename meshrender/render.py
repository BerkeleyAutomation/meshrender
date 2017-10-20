import ctypes
import numpy as np
import trimesh

from .shaders import simple_vertex_shader, simple_fragment_shader
from .light import AmbientLight, PointLight, DirectionalLight
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLX import *
from OpenGL.GL import shaders
from OpenGL.arrays import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PIL import Image

MAX_N_LIGHTS = 10

class OpenGLRenderer(object):

    def __init__(self, scene, smooth=False, size=(500,500)):

        #TODO Set up camera 
        self._scene = scene

        self._meshes = [o['geometry'] for o in scene.objects]
        self._transforms = [o['transform'] for o in scene.objects]
        #TODO Make smoothness a material property
        self._smooth = smooth
        self._size = size
        self._near = 0.05
        self._far = 100.00
        self._P, self._V = self.generate_camera()

        #pygame.init()
        #screen = pygame.display.set_mode((512, 512), pygame.OPENGL|pygame.DOUBLEBUF)
        glutInit()
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(1,1)
        glutInitContextVersion(3,3)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        self._window = glutCreateWindow('render')
        glutHideWindow()

        # Initialize the OpenGL context
        self._colorbuf, self._depthbuf = glGenRenderbuffers(2)
        glBindRenderbuffer(GL_RENDERBUFFER, self._colorbuf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, size[0], size[1])
        glBindRenderbuffer(GL_RENDERBUFFER, self._depthbuf)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size[0], size[1])

        self._framebuf = glGenFramebuffers(1)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._framebuf)
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._colorbuf)
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depthbuf)

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        #glEnable(GL_CULL_FACE)

        self._vaids = self.load_meshes(self._meshes, self._smooth)
        self._shader = self.load_shaders(simple_vertex_shader, simple_fragment_shader)

    def render(self):
        self.draw()
        glutDestroyWindow(self._window)

    def load_shaders(self, vertex_shader, fragment_shader):

        shader = shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )

        return shader

    def generate_camera(self):
        P = self.perspective(np.pi / 4, self._size[0]/self._size[1], self._near, self._far)
        V = self.look_at(0.3*np.array([-1,0,0]), np.zeros(3), np.array([0,1,0]))
        return P, V

    def perspective(self, fovy, aspect, znear, zfar):

        tan_half_fovy = np.tan(fovy / 2.0)
        res = np.zeros((4,4))
        res[0][0] = 1.0 / (aspect * tan_half_fovy)
        res[1][1] = 1.0 / (tan_half_fovy)
        res[2][2] = -(zfar + znear) / (zfar - znear)
        res[3][2] = -1.0
        res[2][3] = -(2.0 * zfar * znear) / (zfar - znear)

        return res

    def look_at(self, eye, center, up):
        f = (center - eye) / np.linalg.norm(center - eye)
        s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
        u = np.cross(s, f)

        x = np.array([
            [s[0], s[1], s[2], -np.dot(s, eye)],
            [u[0], u[1], u[2], -np.dot(u, eye)],
            [-f[0], -f[1], -f[2], np.dot(f, eye)],
            [0, 0, 0, 1]
        ])
        return x

    def load_meshes(self, meshes, smooth=False):
        VA_ids = glGenVertexArrays(len(meshes))

        if len(meshes) == 1:
            VA_ids = [VA_ids]

        buffer_ids = []
        for VA_id, mesh in zip(VA_ids, meshes):
            glBindVertexArray(VA_id)

            if smooth:
                # If smooth is True, we use indexed element arrays and set only one normal per vertex.

                # Set up the vertex VBO
                vertexbuffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glBufferData(GL_ARRAY_BUFFER,
                             4*3*len(mesh.vertices),
                             np.array(mesh.vertices.flatten(), dtype=np.float32),
                             GL_STATIC_DRAW)

                # Set up the normal VBO
                normalbuffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, normalbuffer)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
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
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glBufferData(GL_ARRAY_BUFFER,
                             4*3*3*len(mesh.triangles),
                             np.array(mesh.triangles.flatten(), dtype=np.float32),
                             GL_STATIC_DRAW)

                # Set up the normals
                normalbuffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, normalbuffer)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                normals = np.array([[x,x,x] for x in mesh.face_normals], dtype=np.float32)
                normals = normals.flatten()
                glBufferData(GL_ARRAY_BUFFER,
                             4*len(normals),
                             normals,
                             GL_STATIC_DRAW)

            # Unbind all buffers
            glBindVertexArray(0)
            glDisableVertexAttribArray(0)
            glDisableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        return VA_ids

    def draw(self):
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._framebuf)
        glViewport(0,0,self._size[0], self._size[1])

        glClearColor(0.5, 0.5, 0.5, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._shader)

        mvp_id = glGetUniformLocation(self._shader, 'MVP')
        mv_id = glGetUniformLocation(self._shader, 'MV')
        v_id = glGetUniformLocation(self._shader, 'V')
        matprop_id = glGetUniformLocation(self._shader, 'material_properties')
        ambient_id = glGetUniformLocation(self._shader, 'ambient_light_info')
        directional_id = glGetUniformLocation(self._shader, "directional_light_info")
        n_directional_id = glGetUniformLocation(self._shader, "n_directional_lights")
        point_id = glGetUniformLocation(self._shader, "point_light_info")
        n_point_id = glGetUniformLocation(self._shader, "n_point_lights")

        glUniformMatrix4fv(v_id, 1, GL_TRUE, self._V)

        glUniform4fv(matprop_id, 1, np.array([0.6, 0.3, 1.0, 4.0]))

        scene = self._scene

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

        for vaid, mesh, M in zip(self._vaids, self._meshes, self._transforms):
            glBindVertexArray(vaid)

            MVP = self._P.dot(self._V.dot(M))
            glUniformMatrix4fv(mvp_id, 1, GL_TRUE, MVP)
            glUniformMatrix4fv(mv_id, 1, GL_TRUE, self._V.dot(M))

            if self._smooth:
                glDrawElements(GL_TRIANGLES, 3*len(mesh.faces), GL_UNSIGNED_INT, ctypes.c_void_p(0))
            else:
                glDrawArrays(GL_TRIANGLES, 0, 3*len(mesh.faces))

            glBindVertexArray(0)

        glUseProgram(0)

        glFlush()

        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._framebuf)
        buf = (GLubyte * (3 * self._size[0] * self._size[1]))(0)
        glReadPixels(0,0,self._size[0], self._size[1], GL_RGB, GL_UNSIGNED_BYTE, buf)
        z = (GLfloat * (self._size[0] * self._size[1]))(0)
        glReadPixels(0,0,self._size[0], self._size[1], GL_DEPTH_COMPONENT, GL_FLOAT, z)

        x = np.frombuffer(z, dtype=np.float32).reshape(self._size)
        x = np.flip(x, axis=0)
        x = 2.0 * x - 1.0
        x = 2.0 * self._near * self._far / (self._far + self._near - x * (self._far - self._near))

        img = Image.frombytes(mode='RGB', size=self._size, data=buf)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        img.save('shit.jpg')

        img = Image.fromarray(x)
        img = img.convert('RGB')

        img.save('shit1.jpg')

