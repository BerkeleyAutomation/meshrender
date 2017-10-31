import copy

import pyglet
import pyglet.gl as gl

import numpy as np

from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import *
from OpenGL.GLUT import *

from .constants import MAX_N_LIGHTS
from .light import AmbientLight, PointLight, DirectionalLight
from .shaders import vertex_shader, fragment_shader
from .camera import VirtualCamera

from autolab_core import transformations
from autolab_core import RigidTransform
from perception import CameraIntrinsics

class Trackball(object):
    STATE_ROTATE = 0
    STATE_PAN = 1
    STATE_ROLL = 2
    STATE_ZOOM = 3

    def __init__(self, T_camera_world,
                 target=np.array([0.0, 0.0, 0.0]), width=1, height=1, scale=1.0):
        """
        """
        self._T_camera_world = T_camera_world
        self._target = target

        self._n_T_camera_world = T_camera_world
        self._n_target = target

        self._width = width
        self._height = height
        self._scale = scale
        self._state = Trackball.STATE_ROTATE

    def set_state(self, state):
        """
        """
        self._state = state

    def resize(self, width, height):
        """
        """
        self._width = width
        self._height = height

    def down(self, point):
        """
        """
        self._pdown = point
        self._T_camera_world = self._n_T_camera_world
        self._target = self._n_target

    def drag(self, point):
        """
        """
        dx, dy = point - self._pdown
        dx, dy = float(dx), float(dy)
        mindim = 0.3 * min(self._width, self._height)

        target = self._target
        x_axis = self._T_camera_world.matrix[:3,0].flatten()
        y_axis = self._T_camera_world.matrix[:3,1].flatten()
        z_axis = self._T_camera_world.matrix[:3,2].flatten()
        eye = self._T_camera_world.matrix[:3,3].flatten()

        # Interpret drag as a rotation
        if self._state == Trackball.STATE_ROTATE:
            x_angle = -dx / mindim
            x_rot_mat = transformations.rotation_matrix(x_angle, y_axis, target)
            x_rot_tf = RigidTransform(x_rot_mat[:3,:3], x_rot_mat[:3,3], from_frame='world', to_frame='world')

            y_angle = dy / mindim
            y_rot_mat = transformations.rotation_matrix(y_angle, x_axis, target)
            y_rot_tf = RigidTransform(y_rot_mat[:3,:3], y_rot_mat[:3,3], from_frame='world', to_frame='world')

            self._n_T_camera_world = y_rot_tf.dot(x_rot_tf.dot(self._T_camera_world))

        elif self._state == Trackball.STATE_ROLL:
            center = np.array([self._width / 2.0, self._height / 2.0])
            v_init = self._pdown - center
            v_curr = point - center
            v_init = v_init / np.linalg.norm(v_init)
            v_curr = v_curr / np.linalg.norm(v_curr)

            theta = np.arctan2(v_curr[1], v_curr[0]) - np.arctan2(v_init[1], v_init[0])

            rot_mat = transformations.rotation_matrix(-theta, z_axis, target)
            rot_tf = RigidTransform(rot_mat[:3,:3], rot_mat[:3,3], from_frame='world', to_frame='world')

            self._n_T_camera_world = rot_tf.dot(self._T_camera_world)

        elif self._state == Trackball.STATE_PAN:
            dx = -dx / (5.0*mindim) * self._scale
            dy = -dy / (5.0*mindim) * self._scale

            translation = dx * x_axis + dy * y_axis
            self._n_target = self._target + translation
            t_tf = RigidTransform(translation=translation, from_frame='world', to_frame='world')
            self._n_T_camera_world = t_tf.dot(self._T_camera_world)

        elif self._state == Trackball.STATE_ZOOM:
            radius = np.linalg.norm(eye - target)
            ratio = 0.0
            # Zoom out
            if dy < 0:
                ratio = np.exp(abs(dy)/(0.5*self._height)) - 1.0
            elif dy > 0:
                ratio = 1.0 - np.exp(-dy/(0.5*(self._height)))
            translation = -np.sign(dy) * ratio * radius * z_axis
            t_tf = RigidTransform(translation=translation, from_frame='world', to_frame='world')
            self._n_T_camera_world = t_tf.dot(self._T_camera_world)

    def scroll(self, clicks):
        target = self._target

        ratio = 0.90

        z_axis = self._n_T_camera_world.matrix[:3,2].flatten()
        eye = self._n_T_camera_world.matrix[:3,3].flatten()
        radius = np.linalg.norm(eye - target)
        translation = clicks * (1 - ratio) * radius * z_axis
        t_tf = RigidTransform(translation=translation, from_frame='world', to_frame='world')
        self._n_T_camera_world = t_tf.dot(self._n_T_camera_world)

        z_axis = self._T_camera_world.matrix[:3,2].flatten()
        eye = self._T_camera_world.matrix[:3,3].flatten()
        radius = np.linalg.norm(eye - target)
        translation = clicks * (1 - ratio) * radius * z_axis
        t_tf = RigidTransform(translation=translation, from_frame='world', to_frame='world')
        self._T_camera_world = t_tf.dot(self._T_camera_world)

    def T_camera_world(self):
        """
        """
        return self._n_T_camera_world

class SceneViewer(pyglet.window.Window):

    def __init__(self, scene, size=(640,480)):

        self._scene = scene
        self.scene = scene
        self.bounds = self.compute_scene_bounds()
        self._width = scene.camera.intrinsics.width
        self._height = scene.camera.intrinsics.height
        self.orig_camera = copy.copy(scene.camera)

        try:
            conf = gl.Config(sample_buffers=1,
                             samples=4,
                             depth_size=24,
                             double_buffer=True)
            super(SceneViewer, self).__init__(config=conf,
                                              resizable=True,
                                              width=self._width,
                                              height=self._height)
        except pyglet.window.NoSuchConfigException:
            conf = gl.Config(double_buffer=True)
            super(SceneViewer, self).__init__(config=conf,
                                              resizable=True,
                                              width=self._width,
                                              height=self._height)
        self.trackball = None
        self.reset_view()
        self.init_gl()
        self.update_flags()
        pyglet.app.run()

    def compute_scene_bounds(self):
        """The axis aligned bounds of the scene.

        Returns
        -------
        (2,3) float
            The bounding box with [min, max] coordinates.
        """
        lb = np.array([np.infty, np.infty, np.infty])
        ub = -1.0 * np.array([np.infty, np.infty, np.infty])
        for on in self.scene.objects:
            o = self.scene.objects[on]
            tf_verts = o.T_obj_world.matrix[:3,:3].dot(o.mesh.vertices.T).T + o.T_obj_world.matrix[:3,3]
            lb_mesh = np.min(tf_verts, axis=0)
            ub_mesh = np.max(tf_verts, axis=0)
            lb = np.minimum(lb, lb_mesh)
            ub = np.maximum(ub, ub_mesh)
        return np.array([lb, ub])

        return np.mean(self.compute_scene_bounds(), axis=0)

    def reset_view(self, flags=None):
        centroid = np.mean(self.bounds, axis=0)
        extents = np.diff(self.bounds, axis=0).reshape(-1)
        scale = (extents ** 2).sum() ** .5
        width, height = self._width, self._height

        fov = np.pi / 6
        fl = height / (2.0 * np.tan(fov / 2))
        # Set up reasonable camera intrinsics
        ci = CameraIntrinsics(
            frame = 'camera',
            fx = fl,
            fy = fl,
            cx = width/2.0,
            cy = height/2.0,
            skew=0.0,
            height=height,
            width=width
        )

        # Set up the camera pose (z axis faces away from scene, x to right, y up)
        cp = RigidTransform(
            rotation = np.array([
                [0.0, 0.0, -1.0],
                [0.0, 1.0,  0.0],
                [1.0, 0.0,  0.0]
            ]),
            translation = np.array([-3*scale, 0.0, 0.0]) + centroid,
            from_frame='camera',
            to_frame='world'
        )

        # Create a VirtualCamera
        camera = VirtualCamera(ci, cp)
        self.scene.camera = camera


        self.trackball = Trackball(
            self.scene.camera.T_camera_world,
            target=centroid,
            width=self.scene.camera.intrinsics.width,
            height=self.scene.camera.intrinsics.height,
            scale=scale
        )

        self.view = {'wireframe': False}
        self.trackball.set_state(Trackball.STATE_ROTATE)

        self.update_flags()

    def init_gl(self):
        glClearColor(.93, .93, 1, 1)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self._vaids = self._load_meshes()
        self._shader = self._load_shaders(vertex_shader, fragment_shader)

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
        VA_ids = glGenVertexArrays(len(self._scene.objects))

        if len(self._scene.objects) == 1:
            VA_ids = [VA_ids]

        buffer_ids = []
        for VA_id, obj in zip(VA_ids, self._scene.objects.values()):
            mesh = obj.mesh
            material = obj.material

            glBindVertexArray(VA_id)

            if material.smooth:
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


    def on_draw(self):
        scene = self.scene
        camera = self.scene.camera
        width = camera.intrinsics.width
        height = camera.intrinsics.height

        # Change camera transform using arcball
        #v = _view_transform(self.view)
        #T_world_camera = RigidTransform(v[:3,:3], v[:3,3], from_frame='world', to_frame='camera')
        #T_camera_world = T_world_camera.inverse()
        camera.T_camera_world = self.trackball.T_camera_world()

        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._shader)

        # Get Uniform Locations from Shader
        mvp_id = glGetUniformLocation(self._shader, 'MVP')
        mv_id = glGetUniformLocation(self._shader, 'MV')
        v_id = glGetUniformLocation(self._shader, 'V')
        matprop_id = glGetUniformLocation(self._shader, 'material_properties')
        object_color_id = glGetUniformLocation(self._shader, 'object_color')
        ambient_id = glGetUniformLocation(self._shader, 'ambient_light_info')
        directional_id = glGetUniformLocation(self._shader, "directional_light_info")
        n_directional_id = glGetUniformLocation(self._shader, "n_directional_lights")
        point_id = glGetUniformLocation(self._shader, "point_light_info")
        n_point_id = glGetUniformLocation(self._shader, "n_point_lights")

        # Bind view matrix
        glUniformMatrix4fv(v_id, 1, GL_TRUE, scene.camera.V)

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
            mesh = obj.mesh
            M = obj.T_obj_world.matrix
            material = obj.material

            glBindVertexArray(vaid)

            MV = camera.V.dot(M)
            MVP = camera.P.dot(MV)
            glUniformMatrix4fv(mvp_id, 1, GL_TRUE, MVP)
            glUniformMatrix4fv(mv_id, 1, GL_TRUE, MV)
            glUniform3fv(object_color_id, 1, material.color)
            glUniform4fv(matprop_id, 1, np.array([material.k_a, material.k_d, material.k_s, material.alpha]))

            if material.smooth:
                glDrawElements(GL_TRIANGLES, 3*len(mesh.faces), GL_UNSIGNED_INT, ctypes.c_void_p(0))
            else:
                glDrawArrays(GL_TRIANGLES, 0, 3*len(mesh.faces))

            glBindVertexArray(0)

        glUseProgram(0)

        glFlush()

    def update_flags(self):
        if self.view['wireframe']:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def on_resize(self, width, height):
        self.scene.camera.resize(width, height)
        self.trackball.resize(width, height)
        self.on_draw()

    def on_mouse_press(self, x, y, buttons, modifiers):
        self.trackball.set_state(Trackball.STATE_ROTATE)
        if (buttons == pyglet.window.mouse.LEFT):
            ctrl = (modifiers & pyglet.window.key.MOD_CTRL)
            shift = (modifiers & pyglet.window.key.MOD_SHIFT)
            if (ctrl and shift):
                self.trackball.set_state(Trackball.STATE_ZOOM)
            elif ctrl:
                self.trackball.set_state(Trackball.STATE_ROLL)
            elif shift:
                self.trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == pyglet.window.mouse.MIDDLE):
            self.trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == pyglet.window.mouse.RIGHT):
            self.trackball.set_state(Trackball.STATE_ZOOM)

        self.trackball.down(np.array([x, y]))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.trackball.drag(np.array([x, y]))

    def on_mouse_scroll(self, x, y, dx, dy):
        self.trackball.scroll(dy)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()

    def toggle_wireframe(self):
        self.view['wireframe'] = not self.view['wireframe']
        self.update_flags()

def _view_transform(view):
    '''
    Given a dictionary containing view parameters,
    calculate a transformation matrix.
    '''
    transform = view['ball'].matrix()
    transform[0:3, 3] = view['center']
    transform[0:3, 3] -= np.dot(transform[0:3, 0:3], view['center'])
    transform[0:3, 3] += view['translation'] * view['scale'] * 5.0
    return transform

