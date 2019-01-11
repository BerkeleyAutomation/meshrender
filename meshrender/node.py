import numpy as np
import transformations

class Node(object):

    def __init__(self,
                 name=None,
                 camera=None,
                 children=None,
                 skin=None,
                 matrix=None,
                 mesh=None,
                 rotation=None,
                 scale=None,
                 translation=None,
                 weights=None,
                 light=None):

        # Set defaults
        if children is None:
            children = []

        if matrix is None:
            self._matrix = np.eye(4)
            if rotation is None:
                rotation = np.array([0.0, 0.0, 0.0, 1.0])
            if translation is None:
                translation = np.zeros(3)
            if scale is None:
                scale = np.ones(3)
            self.rotation = rotation
            self.translation = translation
            self.scale = scale
        else:
            self.matrix = matrix

        self.name = name
        self.camera = camera
        self.children = children
        self.skin = skin
        self.mesh = mesh
        self.weights = weights
        self.light = light

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self._matrix = self._matrix_from_trs()

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation = value
        self._matrix = self._matrix_from_trs()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        self._matrix = self._matrix_from_trs()

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value
        self._rotation = self._r_from_matrix()
        self._scale = self._s_from_matrix()
        self._translation = self._t_from_matrix()

    def _r_from_matrix(self):
        return transformations.quaternion_from_matrix(self.matrix)

    def _s_from_matrix(self):
        return np.array([
            np.linalg.norm(self.matrix[:3,0]),
            np.linalg.norm(self.matrix[:3,1]),
            np.linalg.norm(self.matrix[:3,2])
        ])

    def _t_from_matrix(self):
        return self.matrix[:3,3]

    def _matrix_from_trs(self):
        s = np.eye(4)
        s[0,0] = self.scale[0]
        s[1,1] = self.scale[1]
        s[2,2] = self.scale[2]

        r = transformations.quaternion_matrix(self.rotation)

        t = transformations.translation_matrix(self.translation)

        return t.dot(r.dot(s))
