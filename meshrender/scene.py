import numpy as np

from perception import BinaryImage, ColorImage, DepthImage, RgbdImage, GdImage, RenderMode

from .camera import VirtualCamera
from .scene_object import SceneObject
from .material import MaterialProperties
from .light import AmbientLight, PointLight, DirectionalLight
from .constants import MAX_N_LIGHTS
from .render import OpenGLRenderer

class Scene(object):
    """A scene containing objects and lights for 3D OpenGL rendering.
    """

    def __init__(self, bg_color=None):
        self.bg_color = bg_color
        if self.bg_color is None:
            self.bg_color = np.ones(3)

        self._directional_lights = {}
        self._point_lights = {}
        self._spot_lights = {}
        self._objects = {}
        self._camera = None

    def add_object(self, obj, name=None, parent_name=None, pose=None):
        pass

    def remove_object(self, name):
        pass

    def update_object_pose(self, name, pose):
        pass

    def set_camera(self, camera, parent_name=None, pose=None):
        pass

    def update_camera_pose(self, pose):
        pass

    def remove_camera(self):
        pass
