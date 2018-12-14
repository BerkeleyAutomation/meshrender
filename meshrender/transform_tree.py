import numpy as np
import networkx as nx

class TransformTree(object):

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
        # Find children
        for child in self._digraph[frame]:
            #TODO HERE


    def get_pose(self, from_frame, to_frame=None):
        if to_frame is None:
            to_frame = self.base_frame

        # Get path from from_frame to to_frame
        path = self.tree.shortest_path(from_frame, to_frame)

    def set_pose(self, frame):
        pass

