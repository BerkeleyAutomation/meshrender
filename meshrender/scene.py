import numpy as np
import networkx as nx
import trimesh
import uuid

from .scene_object import SceneObject
from .camera import VirtualCamera
from .light import Light, PointLight, DirectionalLight, SpotLight

class Scene(object):
    """Implementation of a scene graph, which contains objects, lights, and a camera
    for 3D OpenGL rendering.
    """

    def __init__(self, bg_color=None):
        self.bg_color = bg_color
        if self.bg_color is None:
            self.bg_color = np.ones(4)
        if len(self.bg_color) == 3:
            self.bg_color = np.hstack((bg_color, 1.0))

        self._objects = {}
        self._lights = {}
        self._camera = None
        self._camera_name = None
        self._name_map = {}
        self._tree = TransformTree()

    @property
    def objects(self):
        """dict : Map from object node names to SceneObject instances.
        """
        return self._objects.copy()

    @property
    def lights(self):
        return self._lights.copy()

    @property
    def point_lights(self):
        return {k : self._lights[k] for k in self._lights if
                    isinstance(self._lights[k], PointLight)}

    @property
    def directional_lights(self):
        return {k : self._lights[k] for k in self._lights if
                    isinstance(self._lights[k], DirectionalLight)}

    @property
    def spot_lights(self):
        return {k : self._lights[k] for k in self._lights if
                    isinstance(self._lights[k], SpotLight)}
    @property
    def camera(self):
        return self._camera

    @property
    def camera_name(self):
        return self._camera_name

    @property
    def bounds(self):
        # Compute corners
        corners = []
        for obj_name in self._objects:
            obj = self._objects[obj_name]
            pose = self.get_pose(obj_name)
            corners_local = trimesh.bounds.corners(obj.bounds)
            corners_world = pose[:3,:3].dot(corners_local.T).T + pose[:3,3]
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

    def add(self, obj, name=None, parent_name=None, pose=None):
        if name in self._lights:
            raise ValueError('Light with name {} already exists!'.format(name))
        if name in self._objects:
            raise ValueError('Object with name {} already exists!'.format(name))

        if name is None:
            if isinstance(obj, VirtualCamera):
                name = 'camera'
            elif isinstance(obj, SceneObject):
                name = 'object_' + uuid.uuid4().hex
            elif isinstance(obj, PointLight):
                name = 'point_light_' + uuid.uuid4().hex
            elif isinstance(obj, DirectionalLight):
                name = 'dir_light_' + uuid.uuid4().hex
            elif isinstance(obj, SpotLight):
                name = 'spot_light_' + uuid.uuid4().hex
        if parent_name is None:
            parent_name = self._tree.base_frame
        if pose is None:
            pose = np.eye(4)
        else:
            pose = pose.copy()

        if isinstance(obj, VirtualCamera):
            self._camera = obj
            self._camera_name = name
        elif isinstance(obj, SceneObject):
            self._objects[name] = obj
        elif isinstance(obj, Light):
            self._lights[name] = obj

        self._tree.add_node(name, parent_name, pose)
        self._name_map[name] = obj

        return name

    def remove(self, name):
        removed_names = self._tree.remove_node(name)
        for n in removed_names:
            if n in self._lights:
                self._lights.pop(n)
            elif n in self._objects:
                self._objects.pop(n)
                self._bounds = None
            elif name == self._camera_name:
                self._camera = None
                self._camera_name = None
            else:
                raise ValueError('Name {} not in scene'.format(name))
            self._name_map.pop(n)

    def get_pose(self, name, to_frame=None):
        if to_frame is None:
            to_frame = self._tree.base_frame
        return self._tree.get_pose(name, to_frame)

    def set_pose(self, name, pose):
        self._tree.set_pose(name, pose.copy())
        if isinstance(self._name_map[name], SceneObject):
            self._bounds = None

class TransformTree(object):
    """Borrowed and adapted from mikedh/trimesh/scene/transforms.py.
    """

    def __init__(self, base_frame='world'):
        self.base_frame = base_frame
        # Digraph contains directed edges from parents to children,
        # plus pose matrices on nodes.
        self._digraph = nx.DiGraph()

        # Udgraph contains undirected edges, for search.
        self._udgraph = nx.Graph()

        self._digraph.add_node(base_frame, pose=np.eye(4))
        self._udgraph.add_node(base_frame)

        self._path_cache = {} # TODO FIGURE OUT CACHING

    def add_node(self, frame, parent_frame=None, pose=None):
        if parent_frame is None:
            parent_frame = self.base_frame
        elif not parent_frame in self._digraph.nodes:
            raise ValueError('Frame {} not in transform tree'.format(parent_frame))

        if pose is None:
            pose = np.eye(4)

        self._digraph.add_node(frame, pose=pose)
        self._udgraph.add_node(frame)
        self._digraph.add_edge(parent_frame, frame)
        self._udgraph.add_edge(parent_frame, frame)

    def remove_node(self, frame):
        removed_children = []
        self._remove_node(frame, removed_children)
        return removed_children

    def _remove_node(self, frame, removed_children):
        # Find children
        for child in self._digraph[frame]:
            self._remove_node(child, removed_children)
        self._digraph.remove_node(frame)
        self._udgraph.remove_node(frame)
        removed_children.append(frame)

    def get_pose(self, from_frame, to_frame=None):
        if to_frame is None:
            to_frame = self.base_frame

        # Get path from from_frame to to_frame
        path = nx.shortest_path(self._udgraph, from_frame, to_frame)

        # Traverse from from_node to to_node
        pose = np.eye(4)
        for i in range(len(path) - 1):
            if self._digraph.has_edge(path[i], path[i+1]):
                # Path[i] is parent of path[i+1], so invert matrix
                matrix = np.linalg.inv(self._digraph.nodes[path[i]]['pose'])
            else:
                matrix = self._digraph.nodes[path[i]]['pose']
            pose = np.dot(pose, matrix)

        return pose

    def set_pose(self, frame, pose):
        self._digraph.nodes[frame]['pose'] = pose
