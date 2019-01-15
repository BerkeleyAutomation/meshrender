"""Scenes, conforming to the glTF 2.0 standards as specified in
https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#reference-scene

Author: Matthew Matl
"""
import numpy as np
import networkx as nx
import trimesh

from .mesh import Mesh
from .camera import Camera
from .light import Light, PointLight, DirectionalLight, SpotLight
from .node import Node
from .utils import format_color_vector

class Scene(object):
    """A hierarchical scene graph.

    Attributes
    ----------
    name : str, optional
        The user-defined name of this object.
    nodes : list of :obj:`Node`
        The set of all nodes in the scene.
    bg_color : (4,) float, optional
        Background color of scene.
    ambient_light : (3,) float, optional
        Color of ambient light.
    """

    def __init__(self,
                 name=None,
                 nodes=None,
                 bg_color=None,
                 ambient_light=None):

        if bg_color is None:
            bg_color = np.ones(4)

        if ambient_light is None:
            ambient_light = np.zeros(3)

        if nodes is None:
            nodes = []

        self.name = name
        self.nodes = set(nodes)
        self.bg_color = bg_color
        self.ambient_light = ambient_light

        self._name_to_nodes = {}
        self._obj_to_nodes = {}
        self._obj_name_to_nodes = {}
        self._mesh_nodes = set()
        self._point_light_nodes = set()
        self._spot_light_nodes = set()
        self._directional_light_nodes = set()
        self._camera_nodes = set()
        self._main_camera_node = None

        # Transform tree
        self._digraph = nx.DiGraph()
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
    def bg_color(self):
        return self._bg_color

    @bg_color.setter
    def bg_color(self, value):
        if value is None:
            value = np.ones(4)
        else:
            value = format_color_vector(value, 4)
        self._bg_color = value

    @property
    def ambient_light(self):
        return self._ambient_light

    @ambient_light.setter
    def ambient_light(self, value):
        if value is None:
            value = np.ones(4)
        else:
            value = format_color_vector(value, 4)
        self._ambient_light = value

    @property
    def meshes(self):
        """set of :obj:`Mesh` : The meshes in the scene.
        """
        return set([n.mesh for n in self.mesh_nodes])

    @property
    def mesh_nodes(self):
        """set of :obj:`Node` : The nodes containing meshes in the scene.
        """
        return self._mesh_nodes

    @property
    def lights(self):
        return self.point_lights | self.spot_lights | self.directional_lights

    @property
    def light_nodes(self):
        """set of :obj:`Node` : The nodes containing lights in the scene.
        """
        return self.point_light_nodes | self.spot_light_nodes | self.directional_light_nodes

    @property
    def point_lights(self):
        return set([n.light for n in self.point_light_nodes])

    @property
    def point_light_nodes(self):
        """set of :obj:`Node` : The nodes containing point lights in the scene.
        """
        return self._point_light_nodes

    @property
    def spot_lights(self):
        return set([n.light for n in self.spot_light_nodes])

    @property
    def spot_light_nodes(self):
        """set of :obj:`Node` : The nodes containing spot lights in the scene.
        """
        return self._spot_light_nodes

    @property
    def directional_lights(self):
        return set([n.light for n in self.directional_light_nodes])

    @property
    def directional_light_nodes(self):
        """set of :obj:`Node` : The nodes containing directional lights in the scene.
        """
        return self._directional_light_nodes

    @property
    def cameras(self):
        return set([n.camera for n in self.camera_nodes])

    @property
    def camera_nodes(self):
        """set of :obj:`Node` : The nodes containing cameras in the scene.
        """
        return self._camera_nodes

    @property
    def main_camera_node(self):
        """set of :obj:`Node` : The node containing the main camera in the
        scene.
        """
        return self._main_camera_node

    @main_camera_node.setter
    def main_camera_node(self, value):
        if value not in self.nodes:
            raise ValueError('New main camera node must already be in scene')
        self._main_camera_node = value

    def add(self, obj, name=None, pose=None, parent_node=None, parent_name=None):
        """Add an object (mesh, light, or camera) to the scene.

        Parameters
        ----------
        obj : :obj:`Mesh`, :obj:`Light`, or :obj:`Camera`
            The object to add to the scene.
        name : str
            A name for the new node to be created.
        pose : (4,4) float
            The local pose of this node relative to its parent node.
        parent_node : :obj:`Node`
            The parent of this Node. If None, the new node is a root node.
        parent_name : str
            The name of the parent node, can be specified instead of
            `parent_node`.

        Returns
        -------
        node : :obj:`Node`
            The newly-created and inserted node.
        """
        if isinstance(obj, Mesh):
            node = Node(name=name, matrix=pose, mesh=obj)
        elif isinstance(obj, Light):
            node = Node(name=name, matrix=pose, light=obj)
        elif isinstance(obj, Camera):
            node = Node(name=name, matrix=pose, camera=obj)
            if self._main_camera_node is None:
                self._main_camera_node = node
        else:
            raise TypeError('Unrecognized object type')

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
        """Search for an existing node. Only a node matching all specified
        parameters is returned, or None if no such node exists.

        Parameters
        ----------
        node : :obj:`Node`, optional
            If present, this object is simply returned.
        name : str
            A name for the Node.
        obj : :obj:`Mesh`, :obj:`Light`, or :obj:`Camera`
            An object that is attached to the node.
        obj_name : str
            The name of an object that is attached to the node.
        """
        nodes = list(self._get_nodes(node=node, name=name, obj=obj, obj_name=obj_name))
        if len(nodes) == 0:
            return None
        elif len(nodes) > 1:
            raise ValueError('Non-unique node query')
        else:
            return nodes[0]

    def add_node(self, node, parent_node=None):
        """Add a Node to the scene.

        Parameters
        ----------
        node : :obj:`Node`
            The node to be added.
        parent_node : :obj:`Node`
            The parent of this Node. If None, the new node is a root node.
        """
        self.nodes.add(node)
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
            if self._main_camera_node is None:
                self._main_camera_node = node

        # Connect to parent
        if parent_node is None:
            parent_node = 'world'
        self._digraph.add_edge(node, parent_node)

        # Iterate over children
        for child in node.children:
            self.add_node(child, node)

        self._path_cache = {}

    def has_node(self, node):
        """Check if a node is already in the scene.

        Parameters
        ----------
        node : :obj:`Node`
            The node to be checked.

        Returns
        -------
        has_node : bool
            True if the node is already in the scene and false otherwise.
        """
        return node in self.nodes

    def remove_node(self, node):
        """Remove a node and all its children from the scene.

        Parameters
        ----------
        node : :obj:`Node`
            The node to be removed.
        """
        self.nodes.remove(node)
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
            if self._main_camera_node == node:
                if len(self._camera_nodes) > 0:
                    self._main_camera_node = next(iter(self._camera_nodes))
                else:
                    self._main_camera_node = None

        self._path_cache = {}

    def get_pose(self, node):
        """Get the world-frame pose of a node in the scene.

        Parameters
        ----------
        node : :obj:`Node`
            The node to find the pose of.

        Returns
        -------
        pose : (4,4) float
            The transform matrix for this node.
        """
        # Get path from from_frame to to_frame
        path = nx.shortest_path(self._digraph, node, 'world')
        self._path_cache[node] = path

        # Traverse from from_node to to_node
        pose = np.eye(4)
        for n in path[:-1]:
            pose = np.dot(n.matrix, pose)

        return pose

    @property
    def bounds(self):
        """(2,3) float : The axis-aligned bounds of the scene.
        """
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
        """(3,) float : The centroid of the scene's axis-aligned bounding box
        (AABB).
        """
        return np.mean(self.bounds, axis=0)

    @property
    def extents(self):
        """(3,) float : The lengths of the axes of the scene's AABB.
        """
        return np.diff(self.bounds, axis=0).reshape(-1)

    @property
    def scale(self):
        """(3,) float : The length of the diagonal of the scene's AABB.
        """
        return np.linalg.norm(self.extents)
