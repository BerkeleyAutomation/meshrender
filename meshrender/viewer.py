import copy

import pyglet
pyglet.options['shadow_window'] = False
import pyglet.gl as gl
from pyglet import clock

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
    """A trackball class for creating camera transformations from mouse movements.
    """
    STATE_ROTATE = 0
    STATE_PAN = 1
    STATE_ROLL = 2
    STATE_ZOOM = 3

    def __init__(self, T_camera_world, size, scale,
                 target=np.array([0.0, 0.0, 0.0])):
        """Initialize a trackball with an initial camera-to-world pose
        and the given parameters.

        Parameters
        ----------
        T_camera_world : autolab_core.RigidTransform
            An initial camera-to-world pose for the trackball.

        size : (float, float)
            The width and height of the camera image in pixels.

        scale : float
            The diagonal of the scene's bounding box --
            used for ensuring translation motions are sufficiently
            fast for differently-sized scenes.

        target : (3,) float
            The center of the scene in world coordinates.
            The trackball will revolve around this point.
        """
        self._size = np.array(size)
        self._scale = float(scale)

        self._T_camera_world = T_camera_world
        self._n_T_camera_world = T_camera_world

        self._target = target
        self._n_target = target

        self._state = Trackball.STATE_ROTATE

    @property
    def T_camera_world(self):
        """autolab_core.RigidTransform : The current camera-to-world pose.
        """
        return self._n_T_camera_world

    def set_state(self, state):
        """Set the state of the trackball in order to change the effect of dragging motions.

        Parameters
        ----------
        state : int
            One of Trackball.STATE_ROTATE, Trackball.STATE_PAN, Trackball.STATE_ROLL, and
            Trackball.STATE_ZOOM.
        """
        self._state = state

    def resize(self, size):
        """Resize the window.

        Parameters
        ----------
        size : (float, float)
            The new width and height of the camera image in pixels.
        """
        self._size = np.array(size)

    def down(self, point):
        """Record an initial mouse press at a given point.

        Parameters
        ----------
        point : (2,) int
            The x and y pixel coordinates of the mouse press.
        """
        self._pdown = np.array(point, dtype=np.float32)
        self._T_camera_world = self._n_T_camera_world
        self._target = self._n_target

    def drag(self, point):
        """Update the tracball during a drag.

        Parameters
        ----------
        point : (2,) int
            The current x and y pixel coordinates of the mouse during a drag.
            This will compute a movement for the trackball with the relative motion
            between this point and the one marked by down().
        """
        point = np.array(point, dtype=np.float32)
        dx, dy = point - self._pdown
        mindim = 0.3 * np.min(self._size)

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

        # Interpret drag as a roll about the camera axis
        elif self._state == Trackball.STATE_ROLL:
            center = self._size / 2.0
            v_init = self._pdown - center
            v_curr = point - center
            v_init = v_init / np.linalg.norm(v_init)
            v_curr = v_curr / np.linalg.norm(v_curr)

            theta = np.arctan2(v_curr[1], v_curr[0]) - np.arctan2(v_init[1], v_init[0])

            rot_mat = transformations.rotation_matrix(-theta, z_axis, target)
            rot_tf = RigidTransform(rot_mat[:3,:3], rot_mat[:3,3], from_frame='world', to_frame='world')

            self._n_T_camera_world = rot_tf.dot(self._T_camera_world)

        # Interpret drag as a camera pan in view plane
        elif self._state == Trackball.STATE_PAN:
            dx = -dx / (5.0*mindim) * self._scale
            dy = -dy / (5.0*mindim) * self._scale

            translation = dx * x_axis + dy * y_axis
            self._n_target = self._target + translation
            t_tf = RigidTransform(translation=translation, from_frame='world', to_frame='world')
            self._n_T_camera_world = t_tf.dot(self._T_camera_world)

        # Interpret drag as a zoom motion
        elif self._state == Trackball.STATE_ZOOM:
            radius = np.linalg.norm(eye - target)
            ratio = 0.0
            if dy < 0:
                ratio = np.exp(abs(dy)/(0.5*self._size[1])) - 1.0
            elif dy > 0:
                ratio = 1.0 - np.exp(-dy/(0.5*(self._size[1])))
            translation = -np.sign(dy) * ratio * radius * z_axis
            t_tf = RigidTransform(translation=translation, from_frame='world', to_frame='world')
            self._n_T_camera_world = t_tf.dot(self._T_camera_world)

    def scroll(self, clicks):
        """Zoom using a mouse scroll wheel motion.

        Parameters
        ----------
        clicks : int
            The number of clicks. Positive numbers indicate forward wheel movement.
        """
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

    def rotate(self, azimuth, axis=None):
        """Rotate the trackball about the "Up" axis by azimuth radians.

        Parameters
        ----------
        azimuth : float
            The number of radians to rotate.
        """
        target = self._target

        y_axis = self._n_T_camera_world.matrix[:3,1].flatten()
        if axis is not None:
            y_axis = axis
        x_rot_mat = transformations.rotation_matrix(azimuth, y_axis, target)
        x_rot_tf = RigidTransform(x_rot_mat[:3,:3], x_rot_mat[:3,3], from_frame='world', to_frame='world')
        self._n_T_camera_world = x_rot_tf.dot(self._n_T_camera_world)

        y_axis = self._T_camera_world.matrix[:3,1].flatten()
        if axis is not None:
            y_axis = axis
        x_rot_mat = transformations.rotation_matrix(azimuth, y_axis, target)
        x_rot_tf = RigidTransform(x_rot_mat[:3,:3], x_rot_mat[:3,3], from_frame='world', to_frame='world')
        self._T_camera_world = x_rot_tf.dot(self._T_camera_world)


class SceneViewer(pyglet.window.Window):
    """An interactive viewer for a 3D scene.

    This doesn't use the scene's camera - instead, it uses one based on a trackball.

    The basic commands for moving about the scene are given as follows:

    * To rotate the camera about the center of the scene, hold the left mouse button and drag the cursor.
    * To rotate the camera about its viewing axis, hold CTRL and then hold the left mouse button and drag the cursor.
    * To pan the camera, do one of the following:

        * Hold SHIFT, then hold the left mouse button and drag the cursor.
        * Hold the middle mouse button and drag the cursor.

    * To zoom the camera in or our, do one of the following:

        * Scroll the mouse wheel.
        * Hold the right mouse button and drag the cursor.

    Other keyboard commands are as follows:

    * z -- resets the view to the original view.
    * w -- toggles wireframe mode for each mesh in the scene.
    * l -- toggles between raymond lighting and the scene's default lights.
    * a -- toggles rotational animation.
    * n -- toggles 'bad normals' mode.
    """

    def __init__(self, scene, size=(640,480), **flags):
        """Initialize a scene viewer and open the viewer window.

        Parameters
        ----------
        scene : Scene
            A scene to view. The scene's camera is not used.
        size : (int, int)
            The width and height of the target window in pixels.
        flags : other kwargs
            Other optional keyword arguments.

        Note
        ----
        The possible set of flags are given in the Other Parameters section.

        Other Parameters
        ----------------
        line_width : float
            Sets the line width for wireframe meshes (default is 1).
        flip_wireframe : bool
            If True, inverts the 'wireframe' material property for all meshes.
        raymond_lighting : bool
            If True, the scene's point and directional lights are discarded in favor
            of a set of three directional lights that move with the camera.
        front_and_back : bool
            If True, the shader will treat all normals as facing the camera.
        animate : bool
            If True, the camera will rotate by default about the scene.
        animate_az : float
            The number of radians to rotate per timestep.
        animate_rate : float
            The framerate for animation in fps.
        animate_axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        """
        self._gl_initialized = False

        self._scene = scene
        self._size = np.array(size)
        self._camera = None
        self._trackball = None
        self._flags = SceneViewer.default_flags()
        self._flags.update(flags)
        self._raymond_lights = self._create_raymond_lights()
        self._reset_view()

        # Close the scene's window if active
        self._scene.close_renderer()

        # Set up the window
        try:
            conf = gl.Config(sample_buffers=1,
                             samples=4,
                             depth_size=24,
                             double_buffer=True,
                             major_version=3,
                             minor_version=2)
            super(SceneViewer, self).__init__(config=conf,
                                              resizable=True,
                                              width=self._size[0],
                                              height=self._size[1])
        except pyglet.window.NoSuchConfigException:
            conf = gl.Config(double_buffer=True)
            super(SceneViewer, self).__init__(config=conf,
                                              resizable=True,
                                              width=self._size[0],
                                              height=self._size[1])
        self._init_gl()
        self._update_flags()
        pyglet.app.run()

    def on_draw(self):
        """Redraw the scene into the viewing window.
        """
        if not self._gl_initialized:
            return
        scene = self._scene
        camera = self._camera

        camera.T_camera_world = self._trackball.T_camera_world

        # Set viewport size
        context = self.context
        back_width, back_height = self._size

        # Check for retina slash high-dpi displays (hack)
        if hasattr(self.context, '_nscontext'):
            view = self.context._nscontext.view()
            bounds = view.convertRectToBacking_(view.bounds()).size
            back_width, back_height = (int(bounds.width), int(bounds.height))

        glViewport(0, 0, back_width, back_height)
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
        front_and_back_id = glGetUniformLocation(self._shader, "front_and_back")

        # Bind bad normals id
        front_and_back = 0
        if self._flags['front_and_back']:
            front_and_back = 1
        glUniform1i(front_and_back_id, front_and_back)

        # Bind view matrix
        glUniformMatrix4fv(v_id, 1, GL_TRUE, camera.V)

        # Bind ambient lighting
        glUniform4fv(ambient_id, 1, np.hstack((scene.ambient_light.color,
                                               scene.ambient_light.strength)))

        # If using raymond lighting, don't use scene's directional or point lights
        d_lights = scene.directional_lights
        p_lights = scene.point_lights
        if self._flags['raymond_lighting']:
            d_lights = []
            for dlight in self._raymond_lights:
                direc = dlight.direction
                direc = camera.T_camera_world.matrix[:3,:3].dot(direc)
                d_lights.append(DirectionalLight(
                    direction=direc,
                    color=dlight.color,
                    strength=dlight.strength
                ))
            p_lights = []

        # Bind directional lighting
        glUniform1i(n_directional_id, len(d_lights))
        directional_info = np.zeros((2*MAX_N_LIGHTS, 4))
        for i, dlight in enumerate(d_lights):
            directional_info[2*i,:] = np.hstack((dlight.color, dlight.strength))
            directional_info[2*i+1,:] = np.hstack((dlight.direction, 0))
        glUniform4fv(directional_id, 2*MAX_N_LIGHTS, directional_info.flatten())

        # Bind point lighting
        glUniform1i(n_point_id, len(p_lights))
        point_info = np.zeros((2*MAX_N_LIGHTS, 4))
        for i, plight in enumerate(p_lights):
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

            wf = material.wireframe != self._flags['flip_wireframe']
            if wf:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            if material.smooth:
                glDrawElements(GL_TRIANGLES, 3*len(mesh.faces), GL_UNSIGNED_INT, ctypes.c_void_p(0))
            else:
                glDrawArrays(GL_TRIANGLES, 0, 3*len(mesh.faces))

            glBindVertexArray(0)

        glUseProgram(0)

    def on_resize(self, width, height):
        """Resize the camera and trackball when the window is resized.
        """
        self._size = (width, height)
        self._camera.resize(width, height)
        self._trackball.resize(self._size)
        self.on_draw()

    def on_mouse_press(self, x, y, buttons, modifiers):
        """Record an initial mouse press.
        """
        self._trackball.set_state(Trackball.STATE_ROTATE)
        if (buttons == pyglet.window.mouse.LEFT):
            ctrl = (modifiers & pyglet.window.key.MOD_CTRL)
            shift = (modifiers & pyglet.window.key.MOD_SHIFT)
            if (ctrl and shift):
                self._trackball.set_state(Trackball.STATE_ZOOM)
            elif ctrl:
                self._trackball.set_state(Trackball.STATE_ROLL)
            elif shift:
                self._trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == pyglet.window.mouse.MIDDLE):
            self._trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == pyglet.window.mouse.RIGHT):
            self._trackball.set_state(Trackball.STATE_ZOOM)

        self._trackball.down(np.array([x, y]))

        # Stop animating while using the mouse
        clock.unschedule(SceneViewer.animate)


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Record a mouse drag.
        """
        self._trackball.drag(np.array([x, y]))

    def on_mouse_release(self, x, y, button, modifiers):
        """Record a mouse release.
        """
        self._update_flags()

    def on_mouse_scroll(self, x, y, dx, dy):
        """Record a mouse scroll.
        """
        self._trackball.scroll(dy)

    def on_key_press(self, symbol, modifiers):
        """Record a key press.
        """
        if symbol == pyglet.window.key.W:
            self._flags['flip_wireframe'] = not self._flags['flip_wireframe']
        elif symbol == pyglet.window.key.Z:
            self._reset_view()
        elif symbol == pyglet.window.key.L:
            self._flags['raymond_lighting'] = not self._flags['raymond_lighting']
        elif symbol == pyglet.window.key.A:
            self._flags['animate'] = not self._flags['animate']
        elif symbol == pyglet.window.key.N:
            self._flags['front_and_back'] = not self._flags['front_and_back']
        self._update_flags()

    def _update_flags(self):
        """Update OpenGL state based on the current flags.
        """
        glLineWidth(float(self._flags['line_width']))
        clock.unschedule(SceneViewer.animate)
        if self._flags['animate']:
            clock.schedule_interval(SceneViewer.animate, 1.0/self._flags['animate_rate'], self)

    def _reset_view(self):
        """Reset the view to a good initial state.

        The view is initially along the positive x-axis a sufficient distance from the scene.
        """

        # Compute scene bounds and scale
        bounds = self._compute_scene_bounds()
        centroid = np.mean(bounds, axis=0)
        extents = np.diff(bounds, axis=0).reshape(-1)
        scale = (extents ** 2).sum() ** .5
        width, height = self._size

        # Set up reasonable camera intrinsics
        fov = np.pi / 6.0
        fl = height / (2.0 * np.tan(fov / 2))
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
        s2 = 1.0/np.sqrt(2.0)
        cp = RigidTransform(
            rotation = np.array([
                [0.0, -s2,  s2],
                [1.0, 0.0, 0.0],
                [0.0,  s2,  s2]
            ]),
            translation = np.sqrt(2.0)*np.array([scale, 0.0, scale]) + centroid,
            from_frame='camera',
            to_frame='world'
        )

        # Create a VirtualCamera
        self._camera = VirtualCamera(ci, cp)

        # Create a trackball
        self._trackball = Trackball(
            self._camera.T_camera_world,
            self._size, scale,
            target=centroid,
        )

    def _create_raymond_lights(self):
        """Create raymond lights for the scene.
        """
        raymond_lights = []

        # Create raymond lights
        elevs = np.pi * np.array([1/6., 1/6., 1/4.])
        azims = np.pi * np.array([1/6., 5/3., -1/4.])
        l = 0
        for az, el in zip(azims, elevs):
            x = np.cos(el) * np.cos(az)
            y = np.cos(el) * np.sin(el)
            z = np.sin(el)

            direction = -np.array([x, y, z])
            direction = direction / np.linalg.norm(direction)
            direc = DirectionalLight(
                direction=direction,
                color=np.array([1.0, 1.0, 1.0]),
                strength=1.0
            )
            raymond_lights.append(direc)
        return raymond_lights

    def _init_gl(self):
        """Initialize OpenGL by loading shaders and mesh geometry.
        """
        bg = self._scene.background_color
        glClearColor(bg[0], bg[1], bg[2], 1.0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self._vaids = self._load_meshes()
        glBindVertexArray(self._vaids[0])
        self._shader = self._load_shaders(vertex_shader, fragment_shader)
        glBindVertexArray(0)

        self._gl_initialized = True

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
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        return VA_ids

    def _compute_scene_bounds(self):
        """The axis aligned bounds of the scene.

        Returns
        -------
        (2,3) float
            The bounding box with [min, max] coordinates.
        """
        lb = np.array([np.infty, np.infty, np.infty])
        ub = -1.0 * np.array([np.infty, np.infty, np.infty])
        for on in self._scene.objects:
            o = self._scene.objects[on]
            tf_verts = o.T_obj_world.matrix[:3,:3].dot(o.mesh.vertices.T).T + o.T_obj_world.matrix[:3,3]
            lb_mesh = np.min(tf_verts, axis=0)
            ub_mesh = np.max(tf_verts, axis=0)
            lb = np.minimum(lb, lb_mesh)
            ub = np.maximum(ub, ub_mesh)
        return np.array([lb, ub])

    @staticmethod
    def default_flags():
        flags = {
            'flip_wireframe' : False,
            'raymond_lighting' : False,
            'line_width': 1.0,
            'front_and_back': False,
            'animate' : False,
            'animate_az' : 0.05,
            'animate_rate' : 30,
            'animate_axis' : None
        }
        return flags

    @staticmethod
    def animate(dt, self):
        """Animate the scene by rotating the camera.
        """
        self._trackball.rotate(self._flags['animate_az'], self._flags['animate_axis'])


