"""A virtual camera in a 3D scene.
"""
import numpy as np

from perception import CameraIntrinsics
from autolab_core import RigidTransform

from .constants import Z_NEAR, Z_FAR

class VirtualCamera(object):
    """A virtual camera, including its intrinsics and its pose.
    """

    def __init__(self, intrinsics, T_camera_world=RigidTransform(from_frame='camera', to_frame='world')):
        """Initialize a virtual camera with the given intrinsics and initial pose in the world.

        Parameters
        ----------
        intrinsics : percetion.CameraIntrinsics
            The intrinsic properties of the camera, from the Berkeley AUTOLab's perception module.
        T_camera_world : autolab_core.RigidTransform
            A transform from camera to world coordinates that indicates
            the camera's pose. The camera frame's x axis points right,
            its y axis points up, and its negative z axis points towards
            the scene. The RigidTransform object is from the Berkeley AUTOLab's autolab_core module.
        """
        if not isinstance(intrinsics, CameraIntrinsics):
            raise ValueError('intrinsics must be an object of type CameraIntrinsics')

        self._intrinsics = intrinsics
        self.T_camera_world = T_camera_world

        # Compute an OpenGL projection matrix given the camera intrinsics
        P = np.zeros((4,4))
        P[0][0] = 2.0 * intrinsics.fx / intrinsics.width
        P[1][1] = 2.0 * intrinsics.fy / intrinsics.height
        P[0][2] = 1.0 - 2.0 * intrinsics.cx / intrinsics.width
        P[1][2] = 2.0 * intrinsics.cy / intrinsics.height - 1.0
        P[2][2] = -(Z_FAR + Z_NEAR) / (Z_FAR - Z_NEAR)
        P[3][2] = -1.0
        P[2][3] = -(2.0 * Z_FAR * Z_NEAR) / (Z_FAR - Z_NEAR)
        self._P = P

    @property
    def intrinsics(self):
        """perception.CameraIntrinsics: The camera's intrinsic parameters.
        """
        return self._intrinsics

    @property
    def T_camera_world(self):
        """autolab_core.RigidTransform: The camera's pose relative to the world frame.
        """
        return self._T_camera_world

    @T_camera_world.setter
    def T_camera_world(self, T):
        if not isinstance(T, RigidTransform):
            raise ValueError('transform must be an object of type RigidTransform')
        if not T.from_frame == self._intrinsics.frame or not T.to_frame == 'world':
            raise ValueError('transform must be from {} -> world, got {} -> {}'.format(self._intrinsics.frame, T.from_frame, T.to_frame))
        self._T_camera_world = T

    @property
    def V(self):
        """(4,4) float: A homogenous rigid transform matrix mapping world coordinates
        to camera coordinates. Equivalent to the OpenGL View matrix.
        """
        return self.T_camera_world.inverse().matrix

    @property
    def P(self):
        """(4,4) float: A homogenous projective matrix for the camera, equivalent
        to the OpenGL Projection matrix.
        """
        return self._P

