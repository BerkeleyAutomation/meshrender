import numpy as np
import networkx as nx

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

        self._path_cache = {}
        self._tf_cache = {}

    def add_node(self, frame, parent_frame=None, pose=None):
        if parent_frame is None:
            parent_frame = self.base_frame
        elif not parent_frame in self._digraph.nodes:
            raise ValueError('Frame {} not in transform tree'.format(parent_frame))

        if pose is None:
            pose = np.eye(4)

        self._digraph.add_node(frame, pose=pose))
        self._udgraph.add_node(frame)
        self._digraph.add_edge(parent_frame, frame)
        self._udgraph.add_edge(parent_frame, frame)

    def remove_node(self, frame):
        removed_children = []
        self._remove_node(frame, removed_children)
        self._path_cache = {}
        self._tf_cache = {}
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

        if (from_frame, to_frame) in self._tf_cache:
            return self._tf_cache[(from_frame, to_frame)]

        # Get path from from_frame to to_frame
        if (from_frame, to_frame) in self._path_cache:
            path = self._path_cache[(from_frame, to_frame)]
        else:
            path = self._udgraph.shortest_path(from_frame, to_frame)
            self._path_cache[(from_frame, to_frame)] = path

        # Traverse from from_node to to_node
        pose = np.eye(4)
        for i in range(len(path) - 1):
            if self._digraph.has_edge(path[i], path[i+1]):
                # Path[i] is parent of path[i+1], so invert matrix
                matrix = np.linalg.inv(self._digraph[path[i]]['pose'])
            else:
                matrix = self._digraph[path[i]]['pose']
            pose = np.dot(pose, matrix)

        self._tf_cache[(from_frame, to_frame)] = pose

        return pose

    def set_pose(self, frame, pose):
        self._digraph[frame]['pose'] = pose
