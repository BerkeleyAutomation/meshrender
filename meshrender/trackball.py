"""Trackball class for 3D manipulation of viewpoints.
"""
import numpy as np

from autolab_core import transformations, RigidTransform

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
            x_angle = dx / mindim
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

            rot_mat = transformations.rotation_matrix(theta, z_axis, target)
            rot_tf = RigidTransform(rot_mat[:3,:3], rot_mat[:3,3], from_frame='world', to_frame='world')

            self._n_T_camera_world = rot_tf.dot(self._T_camera_world)

        # Interpret drag as a camera pan in view plane
        elif self._state == Trackball.STATE_PAN:
            dx = -dx / (5.0*mindim) * self._scale
            dy = dy / (5.0*mindim) * self._scale

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
            translation = np.sign(dy) * ratio * radius * z_axis
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
        x_rot_mat = transformations.rotation_matrix(-azimuth, y_axis, target)
        x_rot_tf = RigidTransform(x_rot_mat[:3,:3], x_rot_mat[:3,3], from_frame='world', to_frame='world')
        self._n_T_camera_world = x_rot_tf.dot(self._n_T_camera_world)

        y_axis = self._T_camera_world.matrix[:3,1].flatten()
        if axis is not None:
            y_axis = axis
        x_rot_mat = transformations.rotation_matrix(-azimuth, y_axis, target)
        x_rot_tf = RigidTransform(x_rot_mat[:3,:3], x_rot_mat[:3,3], from_frame='world', to_frame='world')
        self._T_camera_world = x_rot_tf.dot(self._T_camera_world)
