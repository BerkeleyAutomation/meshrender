import numpy as np
import uuid

from perception import BinaryImage, ColorImage, DepthImage, RgbdImage, GdImage, RenderMode

from .camera import VirtualCamera
from .scene_object import SceneObject, PointCloudSceneObject, MeshSceneObject, InstancedSceneObject
from .material import MaterialProperties
from .light import Light, PointLight, DirectionalLight, SpotLight
from .constants import MAX_N_LIGHTS
from .render import OpenGLRenderer
from .transform_tree import TransformTree

class Scene(object):
    """A scene containing objects and lights for 3D OpenGL rendering.
    """

    def __init__(self, bg_color=None):
        self.bg_color = bg_color
        if self.bg_color is None:
            self.bg_color = np.ones(3)

        self._objects = {}
        self._lights = {}
        self._camera = None
        self._tree = TransformTree()

    def add_object(self, obj, name=None, parent_name=None, pose=None):
        self._add(obj, name, parent_name, pose)

    def remove_object(self, name):
        self._remove(name)

    def get_object_pose(self, name, to_frame=None):
        if to_frame is None:
            to_frame = self._tree.base_frame
        return self._tree.get_pose(name, to_frame)

    def update_object_pose(self, name, pose):
        self._tree.set_pose(name, pose)
        self._bounds = None

    def add_light(self, light, name=None, parent_name=None):
        self._add(light, name, parent_name)

    def remove_light(self, name):
        self._remove(name)

    def get_light_pose(self, name, to_frame=None):
        if to_frame is None:
            to_frame = self._tree.base_frame
        return self._tree.get_pose(name, to_frame)

    def add_camera(self, camera, parent_name=None, pose=None):
        self._add(camera, '__camera__', parent_name, pose)

    def remove_camera(self):
        self._remove('__camera__')

    def update_camera_pose(self, pose):
        self._tree.set_pose('__camera__', pose)

    def get_camera_pose(self, to_frame=None):
        if to_frame is None:
            to_frame = self._tree.base_frame
        return self._tree.get_pose('__camera__', to_frame)

    def _add(self, obj, name=None, parent_name=None, pose=None):
        if name is None:
            name = uuid.uuid4().hex
        if parent_name is None:
            parent_name = self._tree.base_frame
        if pose is None:
            pose = np.eye(4)
        self._tree.add_node(name, parent_name, pose)

        if isinstance(obj, Light):
            self._lights[name] = light
        elif isinstance(obj, SceneObject):
            self._objects[name] = obj
        elif isinstance(obj, VirtualCamera):
            self._camera = obj
        else:
            raise TypeError('Unsupported type {}'.format(obj.__class__.__name__))

    def _remove(self, name):
        removed_names = self._tree.remove_node(name)
        for n in removed_names:
            if n in self._lights:
                self._lights.pop(n)
            elif n in self._objects:
                self._objects.pop(n)
            elif n == '__camera__':
                self._camera = None
        self._bounds = None

    @property
    def bounds(self):
        # Compute corners
        corners = []
        for obj_name in self._objects:
            obj = self._objects[obj_name]
            pose = self.get_object_pose(obj_name)
            corners_local = trimesh.bounds.corners(obj.bounds)
            corners_world = pose[:3,:3].dot(bounds.T).T + pose[:3,3]
            corners.append(corners_world)
        corners = np.vstack(corners)
        bounds = np.array([np.min(corners, axis=0), np.max(corners, axis=0)])

        return bounds

    @property
    def centroid(self):
        return np.mean(self.bounds, axis=0)

    @property
    def extents(self):
        return np.diff(self.bounds, axis=0).reshape(-1)

    @property
    def scale(self):
        return np.linalg.norm(self.extents)
