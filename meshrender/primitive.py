import abc
import numpy as np
import six

from OpenGL.GL import *

from .material import Material, MetallicRoughnessMaterial
from .constants import FLOAT_SZ, UINT_SZ, BufFlags, GLTF
from .utils import format_color_array

class Primitive(object):

    def __init__(self,
                 positions,
                 normals=None,
                 tangents=None,
                 texcoord_0=None,
                 texcoord_1=None,
                 color_0=None,
                 joints_0=None,
                 weights_0=None,
                 indices=None,
                 material=None,
                 mode=None,
                 targets=None,
                 poses=None):

        if mode is None:
            mode = GLTF.TRIANGLES

        self.positions = positions
        self.normals = normals
        self.tangents = tangents
        self.texcoord_0 = texcoord_0
        self.texcoord_1 = texcoord_1
        self.color_0 = color_0
        self.joints_0 = joints_0
        self.weights_0 = weights_0
        self.indices = indices
        self.material = material
        self.mode = mode
        self.targets = targets
        self.poses = poses

        self._bounds = None
        self._vaid = None
        self._buffers = []
        self._is_transparent = None
        self._buf_flags = None

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = np.ascontiguousarray(value.astype(np.float32))
        self._bounds = None

    @property
    def normals(self):
        return self._normals

    @normals.setter
    def normals(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.float32))
            if value.shape != self.positions.shape:
                raise ValueError('Incorrect normals shape')
        self._normals = value

    @property
    def tangents(self):
        return self._tangents

    @tangents.setter
    def tangents(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.float32))
            if value.shape != (self.positions.shape[0], 4):
                raise ValueError('Incorrect tangent shape')
        self._tangents = value

    @property
    def texcoord_0(self):
        return self._texcoord_0

    @texcoord_0.setter
    def texcoord_0(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.float32))
            if value.ndim != 2 or value.shape[0] != self.positions.shape[0] or value.shape[1] < 2:
                raise ValueError('Incorrect texture coordinate shape')
            if value.shape[1] > 2:
                value = value[:,:2]
        self._texcoord_0 = value

    @property
    def texcoord_1(self):
        return self._texcoord_1

    @texcoord_1.setter
    def texcoord_1(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.float32))
            if value.ndim != 2 or value.shape[0] != self.positions.shape[0] or value.shape[1] != 2:
                raise ValueError('Incorrect texture coordinate shape')
        self._texcoord_1 = value

    @property
    def color_0(self):
        return self._color_0

    @color_0.setter
    def color_0(self, value):
        if value is not None:
            value = np.ascontiguousarray(format_color_array(value, 4))
            if value.shape[0] != self.positions.shape[0]:
                raise ValueError('Incorrect vertex color shape')
        self._is_transparent = None
        self._color_0 = value

    @property
    def joints_0(self):
        return self._joints_0

    @joints_0.setter
    def joints_0(self, value):
        self._joints_0 = value

    @property
    def weights_0(self):
        return self._weights_0

    @weights_0.setter
    def weights_0(self, value):
        self._weights_0 = value

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.uint32))
        self._indices = value

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        # Create default material
        if value is None:
            value = MetallicRoughnessMaterial()
        else:
            if not isinstance(value, Material):
                raise ValueError('Object material must be of type Material')
        self._material = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        value = int(value)
        if value < GLTF.POINTS or value > GLTF.TRIANGLE_FAN:
            raise ValueError('Invalid mode')
        self._mode = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    @property
    def poses(self):
        return self._poses

    @poses.setter
    def poses(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.float32))
            if value.ndim == 2:
                value = value[np.newaxis,:,:]
            if value.shape[1] != 4 or value.shape[2] != 4:
                raise ValueError('Incorrect shape of pose matrices, must be (n,4,4)!')
        self._poses = value
        self._bounds = None

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = self._compute_bounds()
        return self._bounds

    @property
    def buf_flags(self):
        if self._buf_flags is None:
            self._buf_flags = self._compute_buf_flags()
        return self._buf_flags

    @property
    def tex_flags(self):
        if self._tex_flags is None:
            self._tex_flags = self._compute_tex_flags()
        return self._tex_flags

    @property
    def is_transparent(self):
        if self._is_transparent is None:
            self._is_transparent = self._compute_transparency()
        return self._is_transparent

    def _add_to_context(self):
        if self._vaid is not None:
            raise ValueError('Mesh is already bound to a context')

        # Generate and bind VAO
        self._vaid = glGenVertexArrays(1)
        glBindVertexArray(self._vaid)

        ########################################################################
        # Fill vertex buffer
        ########################################################################

        # Generate and bind vertex buffer
        vertexbuffer = glGenBuffers(1)
        self._buffers.append(vertexbuffer)
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer)

        # positions
        vertex_data = self.positions
        attr_sizes = [3]

        # Normals
        if self.normals is not None:
            vertex_data = np.hstack((vertex_data, self.normals))
            attr_sizes.append(3)

        # Tangents
        if self.tangents is not None:
            vertex_data = np.hstack((vertex_data, self.tangents))
            attr_sizes.append(4)

        # Texture Coordinates
        if self.texcoord_0 is not None:
            vertex_data = np.hstack((vertex_data, self.texcoord_0))
            attr_sizes.append(2)
        if self.texcoord_1 is not None:
            vertex_data = np.hstack((vertex_data, self.texcoord_1))
            attr_sizes.append(2)

        # Color
        if self.color_0 is not None:
            vertex_data = np.hstack((vertex_data, self.color_0))
            attr_sizes.append(4)

        # TODO JOINTS AND WEIGHTS
        # PASS

        # Copy data to buffer
        vertex_data = np.ascontiguousarray(vertex_data.flatten().astype(np.float32))
        glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ * len(vertex_data), vertex_data, GL_STATIC_DRAW)
        total_sz = sum(attr_sizes)
        offset = 0
        for i, sz in enumerate(attr_sizes):
            glVertexAttribPointer(i, sz, GL_FLOAT, GL_FALSE, FLOAT_SZ * total_sz,
                                  ctypes.c_void_p(FLOAT_SZ * offset))
            glEnableVertexAttribArray(i)
            offset += sz

        ########################################################################
        # Fill model matrix buffer
        ########################################################################

        if self.poses is not None:
            pose_data = np.ascontiguousarray(self.poses.flatten().astype(np.float32))
        else:
            pose_data = np.ascontiguousarray(np.eye(4).flatten().astype(np.float32))

        modelbuffer = glGenBuffers(1)
        self._buffers.append(modelbuffer)
        glBindBuffer(GL_ARRAY_BUFFER, modelbuffer)
        glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ*len(pose_data), pose_data, GL_STATIC_DRAW)

        for i in range(0, 4):
            idx = i + len(attr_sizes)
            glEnableVertexAttribArray(idx)
            glVertexAttribPointer(idx, 4, GL_FLOAT, GL_FALSE, FLOAT_SZ*4, ctypes.c_void_p(4*FLOAT_SZ*i))
            glVertexAttribDivisor(idx, 1);

        ########################################################################
        # Fill element buffer
        ########################################################################
        if self.indices is not None:
            elementbuffer = glGenBuffers(1)
            self._buffers.append(elementbuffer)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, UINT_SZ*self.indices.size,
                         self.indices.flatten().astype(np.uint32), GL_STATIC_DRAW)

        glBindVertexArray(0)

    def _remove_from_context(self):
        if self._vaid is not None:
            glDeleteVertexArrays(1, [self._vaid])
            glDeleteBuffers(len(self._buffers), self._buffers)
            self._vaid = None
            self._buffers = []

    def _in_context(self):
        return self._vaid is not None

    def _bind(self):
        if self._vaid is None:
            raise ValueError('Cannot bind a Mesh that has not been added to a context')
        glBindVertexArray(self._vaid)

    def _unbind(self):
        glBindVertexArray(0)

    def delete(self):
        self._unbind()
        self._remove_from_context()

    def _compute_bounds(self):
        """Compute the bounds of this object.
        """
        # Compute bounds of this object
        bounds = np.array([np.min(self.positions, axis=0),
                           np.max(self.positions, axis=0)])

        # If instanced, compute translations for approximate bounds
        if self.poses is not None:
            bounds += np.array([np.min(self.poses[:,:3,3], axis=0),
                                np.max(self.poses[:,:3,3], axis=0)])
        return bounds

    def _compute_transparency(self):
        """Compute whether or not this object is transparent.
        """
        if self.color_0 is not None:
            if np.any(self._color_0[:,3] != 1.0):
                return True

        if self.material.is_transparent:
            return True

        return False

    def _compute_buf_flags(self):
        buf_flags = BufFlags.POSITION

        if self.normals is not None:
            buf_flags |= BufFlags.NORMAL
        if self.tangents is not None:
            buf_flags |= BufFlags.TANGENT
        if self.texcoord_0 is not None:
            buf_flags |= BufFlags.TEXCOORD_0
        if self.texcoord_1 is not None:
            buf_flags |= BufFlags.TEXCOORD_1
        if self.color_0 is not None:
            buf_flags |= BufFlags.COLOR_0
        if self.joints_0 is not None:
            buf_flags |= BufFlags.JOINTS_0
        if self.weights_0 is not None:
            buf_flags |= BufFlags.WEIGHTS_0

        return buf_flags
