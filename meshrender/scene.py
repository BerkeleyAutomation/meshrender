import numpy as np
import networkx as nx
import trimesh

from .mesh import Mesh
from .camera import Camera
from .light import Light, PointLight, DirectionalLight, SpotLight
from .node import Node

from .utils import format_color_vector

class Scene(object):
    """Implementation of a scene graph, which contains objects, lights, and a camera
    for 3D OpenGL rendering.
    """

    def __init__(self,
                 name=None,
                 nodes=None,
                 bg_color=None):

        if bg_color is None:
            bg_color = np.ones(4)
        else:
            bg_color = format_color_vector(bg_color, 4)

        if nodes is None:
            nodes = []

        self.name = name
        self.nodes = set(nodes)
        self.bg_color = bg_color


        self._name_to_nodes = {}
        self._obj_to_nodes = {}
        self._obj_name_to_nodes = {}
        self._mesh_nodes = set()
        self._point_light_nodes = set()
        self._spot_light_nodes = set()
        self._directional_light_nodes = set()
        self._camera_nodes = set()

        # Transform tree
        self._digraph = nx.Digraph()
        self._digraph.add_node('world')
        self._path_cache = {}

        # Find root nodes and add them
        if len(nodes) > 0:
            node_parent_map = { n : None for n in nodes }
            for node in nodes:
                for child in node.children:
                    if node_parent_map[child] is not None:
                        raise ValueError('Nodes may not have more than one parent')
                    node_parent_map[child] = node
            root_nodes = [n for n in parent_node_map if parent_node_map[n] is None]
            if len(root_nodes) == 0:
                raise ValueError('Need at least one root node in scene')

            for node in root_nodes:
                self.add_node(node)

    @property
    def meshes(self):
        return set([n.mesh for n in self.mesh_nodes])

    @property
    def mesh_nodes(self):
        return self.mesh_nodes

    @property
    def point_lights(self):
        return set([n.light for n in self.point_light_nodes])

    @property
    def point_light_nodes(self):
        return self._point_light_nodes

    @property
    def spot_lights(self):
        return set([n.light for n in self.spot_light_nodes])

    @property
    def spot_light_nodes(self):
        return self._spot_light_nodes

    @property
    def directional_lights(self):
        return set([n.light for n in self.directional_light_nodes])

    @property
    def directional_light_nodes(self):
        return self._directional_light_nodes

    @property
    def cameras(self):
        return set([n.camera for n in self.camera_nodes])

    @property
    def camera_nodes(self):
        return self._camera_nodes

    def add(self, obj, name=None, pose=None, parent_node=None, parent_name=None):
        if isinstance(obj, Mesh):
            node = Node(name=name, matrix=pose, mesh=obj)
        elif isinstance(obj, Light):
            node = Node(name=name, matrix=pose, light=obj)
        elif isinstance(obj, Camera):
            node = Node(name=name, matrix=pose, camera=obj)

        if parent_node is None and parent_name is not None:
            parent_node = self.get_node(name=parent_name)

        self.add_node(node, parent_node=parent_node)

        return node

    def _get_nodes(self, node=None, name=None, obj=None, obj_name=None):
        if node is not None:
            return set([node])
        nodes = set(self.nodes)
        if name is not None:
            matches = set()
            if name in self._name_to_nodes:
                matches = self._name_to_nodes[name]
            nodes = nodes & matches
        if obj is not None:
            matches = set()
            if obj in self._obj_to_nodes:
                matches = self._obj_to_nodes[obj]
            nodes = nodes & matches
        if obj_name is not None:
            matches = set()
            if obj_name in self._obj_name_to_nodes:
                matches = self._obj_name_to_nodes[obj_name]
            nodes = nodes & matches

        return nodes

    def get_node(self, node=None, name=None, obj=None, obj_name=None):
        nodes = list(self._get_nodes(node=node, name=name, obj=obj, obj_name=obj_name))
        if len(nodes) == 0:
            return None
        elif len(nodes) > 1:
            raise ValueError('Non-unique node query')
        else:
            return nodes[0]

    def add_node(self, node, parent_node=None):
        # Create node in graph
        self._digraph.add_node(node)

        # Add node to sets
        if node.name is not None:
            if node.name not in self._name_to_nodes:
                self._name_to_nodes[node.name] = set()
            self._name_to_nodes.add(node)
        for obj in [node.mesh, node.camera, node.light]:
            if obj is not None:
                self._obj_to_nodes[obj] = node
                if obj.name is not None:
                    if obj.name not in self._obj_name_to_nodes:
                        self._obj_name_to_nodes[obj.name] = set()
                    self._obj_name_to_nodes[obj.name].add(node)
        if node.mesh is not None:
            self._mesh_nodes.add(node)
        if node.light is not None:
            if isinstance(node.light, PointLight):
                self._point_light_nodes.add(node)
            if isinstance(node.light, SpotLight):
                self._spot_light_nodes.add(node)
            if isinstance(node.light, DirectionalLight):
                self._directional_light_nodes.add(node)
        if node.camera is not None:
            self._camera_nodes.add(node)

        # Connect to parent
        if parent_node is None:
            parent_node = 'world'
        self._digraph.add_edge(node, parent_node)

        # Iterate over children
        for child in node.children:
            self.add_node(child, node)

        self._path_cache = {}

    def remove_node(self, node):
        # Remove children
        for child in node.children:
            self.remove_node(child)
        node.children = []

        # Remove self from the graph
        self._digraph.remove_node(node)

        # Remove from maps
        if node.name in self._name_to_nodes:
            self._name_to_nodes[name].remove(node)
            if len(self._name_to_nodes[name]) == 0:
                self._name_to_nodes.pop(name)
        for obj in [node.mesh, node.camera, node.light]:
            if obj is None:
                continue
            self._obj_to_nodes.pop(obj)
            if obj.name is not None:
                self._obj_name_to_nodes[obj.name].remove(node)
                if len(self._obj_name_to_nodes[obj.name]) == 0:
                    self._obj_name_to_nodes.pop(obj.name)
        if node.mesh is not None:
            self._mesh_nodes.remove(node)
        if node.light is not None:
            if isinstance(node.light, PointLight):
                self._point_light_nodes.remove(node)
            if isinstance(node.light, SpotLight):
                self._spot_light_nodes.remove(node)
            if isinstance(node.light, DirectionalLight):
                self._directional_light_nodes.remove(node)
        if node.camera is not None:
            self._camera_nodes.remove(node)

        self._path_cache = {}

    def get_pose(self, node):
        # Get path from from_frame to to_frame
        path = nx.shortest_path(self._digraph, node, 'world')
        self._path_cache[node] = path

        # Traverse from from_node to to_node
        pose = np.eye(4)
        for n in path[-1]:
            pose = np.dot(pose, n.matrix)

        return pose

    @property
    def bounds(self):
        # Compute corners
        corners = []
        for mesh_node in self.mesh_nodes:
            mesh = mesh_node.mesh
            pose = self.get_pose(mesh_node)
            corners_local = trimesh.bounds.corners(mesh.bounds)
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
