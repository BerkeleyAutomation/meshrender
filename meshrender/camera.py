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
            its y axis points down, and its z axis points towards
            the scene (i.e. standard OpenCV coordinates).
        """
        if not isinstance(intrinsics, CameraIntrinsics):
            raise ValueError('intrinsics must be an object of type CameraIntrinsics')

        self._intrinsics = intrinsics
        self.T_camera_world = T_camera_world

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

        Note that the OpenGL camera coordinate system has x to the right, y up, and z away
        from the scene towards the eye!
        """
        T_camera_world_GL = self.T_camera_world.matrix.copy()
        T_camera_world_GL[:3,2] = -T_camera_world_GL[:3,2] # Reverse Z axis
        T_camera_world_GL[:3,1] = -T_camera_world_GL[:3,1] # Reverse Y axis;
        T_world_camera_GL = np.linalg.inv(T_camera_world_GL)
        return T_world_camera_GL

    @property
    def P(self):
        """(4,4) float: A homogenous projective matrix for the camera, equivalent
        to the OpenGL Projection matrix.
        """
        P = np.zeros((4,4))
        P[0][0] = 2.0 * self.intrinsics.fx / self.intrinsics.width
        P[1][1] = 2.0 * self.intrinsics.fy / self.intrinsics.height
        P[0][2] = 1.0 - 2.0 * self.intrinsics.cx / self.intrinsics.width
        P[1][2] = 2.0 * self.intrinsics.cy / self.intrinsics.height - 1.0
        P[2][2] = -(Z_FAR + Z_NEAR) / (Z_FAR - Z_NEAR)
        P[3][2] = -1.0
        P[2][3] = -(2.0 * Z_FAR * Z_NEAR) / (Z_FAR - Z_NEAR)
        return P


    def resize(self, new_width, new_height):
        """Reset the camera intrinsics for a new width and height viewing window.

        Parameters
        ----------
        new_width : int
            The new window width, in pixels.
        new_height : int
            The new window height, in pixels.
        """
        x_scale = float(new_width) / self.intrinsics.width
        y_scale = float(new_height) / self.intrinsics.height
        center_x = float(self.intrinsics.width-1)/2
        center_y = float(self.intrinsics.height-1)/2
        orig_cx_diff = self.intrinsics.cx - center_x
        orig_cy_diff = self.intrinsics.cy - center_y
        scaled_center_x = float(new_width-1) / 2
        scaled_center_y = float(new_height-1) / 2
        cx = scaled_center_x + x_scale * orig_cx_diff
        cy = scaled_center_y + y_scale * orig_cy_diff
        fx = self.intrinsics.fx * x_scale
        fy = self.intrinsics.fy * x_scale
        scaled_intrinsics = CameraIntrinsics(frame=self.intrinsics.frame,
                                             fx=fx, fy=fy, skew=self.intrinsics.skew,
                                             cx=cx, cy=cy, height=new_height, width=new_width)
        self._intrinsics = scaled_intrinsics

