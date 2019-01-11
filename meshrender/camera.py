"""A virtual camera in a 3D scene.
"""
import abc
import numpy as np
import six

from .constants import DEFAULT_Z_NEAR, DEFAULT_Z_FAR

@six.add_metaclass(abc.ABCMeta)
class Camera(object):

    def __init__(self,
                 name=None,
                 znear=DEFAULT_Z_NEAR,
                 zfar=DEFAULT_Z_FAR):
        self.name = name
        self.znear = znear
        self.zfar = zfar

    @property
    def znear(self):
        return self._znear

    @znear.setter
    def znear(self, value):
        if value < 0:
            raise ValueError('z-near must be >= 0.0')
        self._znear = value

    @property
    def zfar(self):
        return self._zfar

    @zfar.setter
    def zfar(self):
        if zfar is None:
            self._zfar = None
            return
        if value <= 0 or value <= self.znear:
            raise ValueError('zfar must be >0 and >znear')
        self._zfar = zfar

    @abc.abstractmethod
    def get_projection_matrix(self, width=None, height=None):
        pass

class PerspectiveCamera(Camera):

    def __init__(self,
                 name=None,
                 znear=DEFAULT_Z_NEAR,
                 zfar=DEFAULT_Z_FAR,
                 yfov=None,
                 aspect_ratio=None):
        super(PerspectiveCamera, self).__init__(
            name=name,
            znear=znear,
            zfar=zfar,
        )

        self.yfov = yfov
        self.aspect_ratio = aspect_ratio

    @property
    def yfov(self):
        return self._yfov

    @yfov.setter(self)
    def yfov(self, value):
        if value <= 0.0:
            raise ValueError('Field of view must be positive')
        self._yfov = float(value)

    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    @aspect_ratio.setter(self)
    def aspect_ratio(self, value):
        if value is None:
            self._aspect_ratio = None
            return
        if value <= 0.0:
            raise ValueError('Aspect ratio must be positive')
        self._aspect_ratio = float(value)

    def get_projection_matrix(self, width=None, height=None):
        aspect_ratio = self.aspect_ratio
        if aspect_ratio is None:
            if width is None or height is None:
                raise ValueError('Aspect ratio of camera must be defined')
            aspect_ratio = float(width) / float(height)

        a = aspect_ratio
        t = np.tan(self.yfov / 2.0)
        n = self.znear
        f = self.zfar

        P = np.zeros((4,4))
        P[0][0] = 1.0 / (a * t)
        P[1][1] = 1.0 / t
        P[3][2] = -1.0

        if f is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * n
        else:
            P[2][2] = (f + n) / (n - f)
            P[2][3] = (2*f*n) / (n - f)

        return P

class OrthographicCamera(Camera):

    def __init__(self,
                 name=None,
                 znear=DEFAULT_Z_NEAR,
                 zfar=DEFAULT_Z_FAR,
                 xmag=None,
                 ymag=None):
        super(PerspectiveCamera, self).__init__(
            name=name,
            znear=znear,
            zfar=zfar
        )

        self.xmag = xmag
        self.ymag = ymag

    @property
    def xmag(self):
        return self._xmag

    @xmag.setter(self)
    def xmag(self, value):
        if value <= 0.0:
            raise ValueError('X magnification must be positive')
        self._xmag = float(value)

    @property
    def ymag(self):
        return self._ymag

    @ymag.setter(self)
    def ymag(self, value):
        if value <= 0.0:
            raise ValueError('Y magnification must be positive')
        self._ymag = float(value)

    def get_projection_matrix(self, width=None, height=None):
        P = np.zeros((4,4))
        P[0][0] = 1.0 / self.xmag
        P[1][1] = 1.0 / self.ymag
        P[2][2] = 2.0 / (n - f)
        P[2][3] = (f + n) / (n - f)
        P[3][3] = 1.0
        return P

#class Camera(object):
#    """A virtual camera, including its intrinsics and its pose.
#
#    Attributes
#    ----------
#    intrinsics : :obj:`percetion.CameraIntrinsics`
#        The intrinsic properties of the camera, from the Berkeley AUTOLab's perception module.
#    znear : float
#        The near-plane clipping distance, in meters.
#    zfar : float
#        The far-plane clipping distance, in meters.
#    """
#
#    def __init__(self, intrinsics, znear=znear, zfar=zfar):
#        self.intrinsics = intrinsics
#        self.znear = znear
#        self.zfar = zfar
#
#    def V(self, pose):
#        """(4,4) float: A homogenous rigid transform matrix mapping world coordinates
#        to camera coordinates. Equivalent to the OpenGL View matrix.
#
#        Parameters
#        ----------
#        pose : (4,4) float
#            A transform from camera to world coordinates that indicates
#            the camera's pose. The camera frame's x axis points right,
#            its y axis points down, and its z axis points towards
#            the scene (i.e. standard OpenCV coordinates).
#
#        Note that the OpenGL camera coordinate system has x to the right, y up, and z away
#        from the scene towards the eye!
#        """
#        # Create inverse V (map from camera to world)
#        V_inv = pose.copy()
#        V_inv[:3,1:3] *= -1 # Reverse Y and Z axes
#
#        # Compute V (map from world to camera
#        V = np.linalg.inv(V_inv)
#        return V
#
#    @property
#    def P(self):
#        """(4,4) float: A homogenous projective matrix for the camera, equivalent
#        to the OpenGL Projection matrix.
#        """
#        P = np.zeros((4,4))
#        P[0][0] = 2.0 * self.intrinsics.fx / self.intrinsics.width
#        P[1][1] = 2.0 * self.intrinsics.fy / self.intrinsics.height
#        P[0][2] = 1.0 - 2.0 * self.intrinsics.cx / self.intrinsics.width
#        P[1][2] = 2.0 * self.intrinsics.cy / self.intrinsics.height - 1.0
#        P[2][2] = -(self.zfar + self.znear) / (self.zfar - self.znear)
#        P[3][2] = -1.0
#        P[2][3] = -(2.0 * self.zfar * self.znear) / (self.zfar - self.znear)
#        return P
#
#
#    def resize(self, new_width, new_height):
#        """Reset the camera intrinsics for a new width and height viewing window.
#
#        Parameters
#        ----------
#        new_width : int
#            The new window width, in pixels.
#        new_height : int
#            The new window height, in pixels.
#        """
#        # Compute X and Y scaling
#        x_scale = float(new_width) / self.intrinsics.width
#        y_scale = float(new_height) / self.intrinsics.height
#
#        # Compute new intrinsics parameters
#        center_x = float(self.intrinsics.width-1)/2
#        center_y = float(self.intrinsics.height-1)/2
#        orig_cx_diff = self.intrinsics.cx - center_x
#        orig_cy_diff = self.intrinsics.cy - center_y
#        scaled_center_x = float(new_width-1) / 2
#        scaled_center_y = float(new_height-1) / 2
#        cx = scaled_center_x + x_scale * orig_cx_diff
#        cy = scaled_center_y + y_scale * orig_cy_diff
#        fx = self.intrinsics.fx * x_scale
#        fy = self.intrinsics.fy * x_scale
#
#        # Create new intrinsics
#        scaled_intrinsics = CameraIntrinsics(frame=self.intrinsics.frame,
#                                             fx=fx, fy=fy, skew=self.intrinsics.skew,
#                                             cx=cx, cy=cy, height=new_height, width=new_width)
#        self.intrinsics = scaled_intrinsics
