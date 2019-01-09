import abc
import numpy as np
import six

from OpenGL.GL import *

from .material import Material

from .constants import VertexBufferFlags, VertexArrayFlags, TextureFlags, FLOAT_SZ, UINT_SZ

@six.add_metaclass(abc.ABCMeta)
class SceneObject(object):
    """An object in a scene.

    This object can be bound to the current OpenGL instance
    and rendered.

    Attributes
    ----------
    vertices : (n,3) float
        Object vertices.
    poses : (n,4,4) float
        If specified, makes this object an instanced object.
        List of poses for each instance relative to object base frame.
    is_visible : bool
        If False, the object will not be rendered.
    is_transparent : bool
        If True, the object contains some transparency.
    in_context : bool
        If True, the object has been loaded into an OpenGL context.
    vertex_buffer_flags : int
        Flags for data included in the vertex buffer.
    vertex_array_flags : int
        Flags for type of data to be rendered for this object.
    texture_flags : int
        Flags for type of texture data available for this object.
    bounds : (2,3) float
        The least and greatest corners of the object's AABB, in its own frame.
    """
    def __init__(self, vertices, is_visible=True, poses=None):
        self.vertices = vertices
        self.poses = poses
        self.is_visible = is_visible
        self._is_transparent = self._compute_transparency()
        self._in_context = False
        self._bounds = None

        # Compute render flags
        self._vertex_buffer_flags = VertexBufferFlags.POSITION
        self._vertex_array_flags = VertexArrayFlags.POINTS
        self._texture_flags = TextureFlags.NONE
        self._update_flags()

    @property
    def is_transparent(self):
        return self._is_transparent

    @property
    def in_context(self):
        return self._in_context

    @property
    def vertex_buffer_flags(self):
        return self._vertex_buffer_flags

    @property
    def vertex_array_flags(self):
        return self._vertex_array_flags

    @property
    def texture_flags(self):
        return self._texture_flags

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = self._compute_bounds()
        return self._bounds

    @abc.abstractmethod
    def _add_to_context(self):
        """Add the object to the current OpenGL context.
        """
        pass

    @abc.abstractmethod
    def _remove_from_context(self):
        """Remove the object from the current OpenGL context.
        """
        pass

    @abc.abstractmethod
    def _bind(self):
        """Bind this object in the current OpenGL context.
        """
        pass

    @abc.abstractmethod
    def _unbind(self):
        """Unbind this object in the current OpenGL context.
        """
        pass

    @abc.abstractmethod
    def _compute_transparency(self):
        """Compute whether or not this object is transparent.
        """
        pass

    @abc.abstractmethod
    def _update_flags(self):
        """Compute the shading flags for this object.
        """
        pass

    def _compute_bounds(self):
        """Compute the bounds of this object.
        """
        # Compute bounds of this object
        bounds = np.array([np.min(self.vertices, axis=0),
                            np.max(self.vertices, axis=0)])

        # If instanced, compute translations for approximate bounds
        if self.poses is not None:
            bounds += np.array([np.min(self.poses[:,:3,3], axis=0),
                                np.max(self.poses[:,:3,3], axis=0)])
        return bounds


class PointCloudSceneObject(object):
    """A cloud of points.

    Attributes
    ----------
    vertices : (n,3) float
        Object vertices.
    vertex_colors : (n,3) or (n,4) float
        Colors of each vertex.
    poses : (n,4,4) float
        If specified, makes this object an instanced object.
        List of poses for each instance relative to object base frame.
    is_visible : bool
        If False, the object will not be rendered.
    is_transparent : bool
        If True, the object contains some transparency.
    in_context : bool
        If True, the object has been loaded into an OpenGL context.
    vertex_buffer_flags : int
        Flags for data included in the vertex buffer.
    vertex_array_flags : int
        Flags for type of data to be rendered for this object.
    texture_flags : int
        Flags for type of texture data available for this object.
    bounds : (2,3) float
        The least and greatest corners of the object's AABB, in its own frame.
    """

    def __init__(self, vertices, vertex_colors=None, is_visible=True, poses=None):
        self.vertex_colors = vertex_colors

        if self.vertex_colors is None:
            self.vertex_colors = 0.5 * np.ones(size=self.vertices.size)

        self._vaid = None
        self._buffers = None

        super(PointCloudSceneObject, self).__init__(vertices, is_visible, poses)

    def _add_to_context(self):
        """Add the object to the current OpenGL context.
        """
        if self._in_context:
            raise ValueError('SceneObject is already bound to a context')

        # Generate and bind VAO
        self._vaid = glGenVertexArrays(1)
        glBindVertexArray(self._vaid)

        # Generate and bind vertex buffer
        vertexbuffer = glGenBuffers(1)
        self._buffers.append(vertex_buffer)
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer)

        # Fill vertex buffer with [
        #   - vertex position data (3-float),
        #   - vertex color data (4-float)
        # ]
        vc = self.vertex_colors
        if vc.shape[1] == 3:
            vc = np.concatenate(vc, np.ones((vc.shape[0], vc.shape[1], 1)), axis=2)
        vertbuf = np.hstack((self.vertices, vc)).astype(np.float32).flatten()
        glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ*len(vertbuf), vertbuf, GL_STATIC_DRAW)

        # If the object is instanced, bind the model matrix buffer
        if self.vertex_array_flags & VertexArrayFlags.INSTANCED:
            # Load Transposed 
            pose_data = self.poses.flatten().astype(np.float32)
            modelbuffer = glGenBuffers(1)
            self._buffers.append(modelbuffer)
            glBindBuffer(GL_ARRAY_BUFFER, modelbuffer)
            glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ*len(pose_data), pose_data, GL_STATIC_DRAW)

            for i in range(0, 4):
                idx = i + 2
                glEnableVertexAttribArray(idx)
                glVertexAttribPointer(idx, 4, GL_FLOAT, GL_FALSE, FLOAT_SZ*4, ctypes.c_void_p(4*FLOAT_SZ*i))
                glVertexAttribDivisor(idx, 1);

        glBindVertexArray(0)

        self._in_context = True

    def _remove_from_context(self):
        """Remove the object from the current OpenGL context.
        """
        if self._in_context:
            glDeleteVertexArrays(1, [self._vaid])
            glDeleteBuffers(len(self._buffers), self._buffers)
            self._in_context = False

    def _bind(self):
        """Bind this object in the current OpenGL context and load its shader.
        """
        if not self._in_context:
            raise ValueError('Cannot bind a SceneObject that has not been added to a context')

        # Bind Vertex Arrays
        glBindVertexArray(self._vaid)

    def _unbind(self):
        """Unbind this object in the current OpenGL context.
        """
        if not self._in_context:
            raise ValueError('Cannot unbind a SceneObject that has not been added to a context')

        glBindVertexArray(0)

    def _compute_transparency(self):
        if self.vertex_colors.shape[1] == 4:
            if np.any(self.vertex_colors[:,3] < 1.0):
                return True
        return False

    def _update_flags(self):
        """Compute the shading flags for this object.
        """
        self._vertex_buffer_flags = (VertexBufferFlags.POSITION | VertexBufferFlags.COLOR)
        self._vertex_array_flags = (VertexArrayFlags.POINTS)
        if self.poses is not None:
            self._vertex_array_flags |= VertexArrayFlags.INSTANCED
        self._texture_flags = TextureFlags.NONE

class MeshSceneObject(SceneObject):
    """A triangular mesh.

    Attributes
    ----------
    vertices : (n,3) float
        Object vertices.
    vertex_normals : (n,3) float, optional
        Vertex normals.
    faces : (m,3) int, optional
        Specification of faces, if this object is a triangular mesh.
        These integer indices are indexes into the vertices array.
    face_normals : (m,3) float, optional
        Face normals, optionally specified, required for using face colors
        or non-smooth materials.
    vertex_colors : (n,3) or (n,4), float
        Colors for each vertex, if desired. This is overridden by any texture maps.
    face_colors : (m,3) or (m,4), float
        Colors for each face, if desired. This is overridden by any texture maps or vertex colors.
        Face colors cannot be used with a smooth material.
    material : :obj:`Material`
        The material of the object. If not specified, a default grey material
        will be used.
    texture_coords : (n, 2) float, optional
        Texture coordinates for vertices, if needed.
    poses : (n,4,4) float
        If specified, makes this object an instanced object.
        List of poses for each instance relative to object base frame.
    is_visible : bool
        If False, the object will not be rendered.
    is_transparent : bool
        If True, the object contains some transparency.
    in_context : bool
        If True, the object has been loaded into an OpenGL context.
    vertex_buffer_flags : int
        Flags for data included in the vertex buffer.
    vertex_array_flags : int
        Flags for type of data to be rendered for this object.
    texture_flags : int
        Flags for type of texture data available for this object.
    bounds : (2,3) float
        The least and greatest corners of the object's AABB, in its own frame.
    """

    def __init__(self, vertices, vertex_normals=None, faces=None, face_normals=None,
                 vertex_colors=None, face_colors=None, material=None, texture_coords=None,
                 is_visible=True, poses=None):
        self.vertex_normals = vertex_normals
        self.faces = faces
        self.face_normals = face_normals
        self.vertex_colors = vertex_colors
        self.face_colors = face_colors
        self.material = material
        self.texture_coords = texture_coords

        self._vaid = None
        self._buffers = []

        if self.material is None:
            self.material = Material(
                diffuse=np.array([0.5, 0.5, 0.5, 1.0]),
                specular=np.array([0.1, 0.1, 0.1]),
                shininess=10.0,
                smooth=False,
                wireframe=False,
            )

        if self.material.smooth:
            if face_colors is not None:
                raise ValueError('Cannot use face colors with smooth material')
        else:
            if face_normals is None:
                raise ValueError('Cannot use non-smooth material without face normals')

        if texture_coords is not None:
            if not texture_coords.shape == (self.vertices.shape[0], 2):
                raise ValueError('Incorrect texture coordinate shape: {}'.format(texture_coords.shape))

        if vertex_colors is not None:
            if not vertex_colors.shape[0] == self.vertices.shape[0]:
                raise ValueError('Incorrect vertex colors shape: {}'.format(vertex_colors.shape))

        if face_colors is not None:
            if not face_colors.shape[0] == self.face_colors.shape[0]:
                raise ValueError('Incorrect vertex colors shape: {}'.format(vertex_colors.shape))

        super(MeshSceneObject, self).__init__(vertices, is_visible, poses)

    def _add_to_context(self):
        """Add the object to the current OpenGL context.
        """
        if self._in_context:
            raise ValueError('SceneObject is already bound to a context')

        # Generate and bind VAO
        self._vaid = glGenVertexArrays(1)
        glBindVertexArray(self._vaid)

        # Generate and bind vertex buffer
        vertexbuffer = glGenBuffers(1)
        self._buffers.append(vertexbuffer)
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer)

        # First, fill vertex buffer with positions and normals
        if self.material.smooth:
            vp = self.vertices
            vn = self.vertex_normals
        else:
            vp = self.vertices[self.faces].reshape(3*len(self.faces), 3)
            vn = np.repeat(self.face_normals, 3, axis=0)
        vertex_data = np.hstack((vp, vn))
        attr_sizes = [3,3] # Ordered sizes of attributes

        # Next, add vertex colors if present
        if self.vertex_colors is not None or self.face_colors is not None:
            if self.vertex_colors is not None:
                if self.material.smooth:
                    vc = self.vertex_colors
                else:
                    vc = self.vertex_colors[self.faces].\
                            reshape(3*len(self.faces), self.vertex_colors.shape[1])
            else:
                if self.material.smooth:
                    raise ValueError('Cannot have face colors for smooth mesh')
                else:
                    vc = np.repeat(self.face_colors, 3, axis=0)
            if vc.shape[1] == 3:
                vc = np.concatenate(vc, np.ones((vc.shape[0], vc.shape[1], 1)), axis=2)
            vertex_data = np.hstack((vertex_data, vc))
            attr_sizes.append(4)

        # Next, add texture coordinates if needed
        if self.texture_coords is not None:
            if self.material.smooth:
                vt = self.texture_coords
            else:
                vt = self.texture_coords[self.faces].reshape(3*len(self.faces), 2)
            vertex_data = np.hstack((vertex_data, vt))
            attr_sizes.append(2)

            # Compute and bind the tangent/bitangent vectors if needed for
            # normal mapping
            if self.material.normal is not None:
                raise NotImplementedError('Still need to implement normal mapping')

        # Create vertex buffer data
        vertex_data = vertex_data.flatten().astype(np.float32)
        glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ*len(vertex_data), vertex_data, GL_STATIC_DRAW)
        total_sz = sum(attr_sizes)
        offset = 0
        for i, sz in enumerate(attr_sizes):
            glVertexAttribPointer(i, sz, GL_FLOAT, GL_FALSE, FLOAT_SZ*total_sz, ctypes.c_void_p(FLOAT_SZ*offset))
            glEnableVertexAttribArray(i)
            offset += sz

        # If the object is instanced, bind the model matrix buffer
        if self.vertex_array_flags & VertexArrayFlags.INSTANCED:
            # Load Transposed 
            pose_data = self.poses.flatten().astype(np.float32)
        else:
            pose_data = np.eye(4).flatten().astype(np.float32)
        modelbuffer = glGenBuffers(1)
        self._buffers.append(modelbuffer)
        glBindBuffer(GL_ARRAY_BUFFER, modelbuffer)
        glBufferData(GL_ARRAY_BUFFER, FLOAT_SZ*len(pose_data), pose_data, GL_STATIC_DRAW)

        for i in range(0, 4):
            idx = i + len(attr_sizes)
            glEnableVertexAttribArray(idx)
            glVertexAttribPointer(idx, 4, GL_FLOAT, GL_FALSE, FLOAT_SZ*4, ctypes.c_void_p(4*FLOAT_SZ*i))
            glVertexAttribDivisor(idx, 1);

        # If a smooth mesh, bind the element buffer
        if self.material.smooth:
            elementbuffer = glGenBuffers(1)
            self._buffers.append(elementbuffer)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, UINT_SZ*3*len(mesh.faces),
                         mesh.faces.flatten().astype(np.uint32), GL_STATIC_DRAW)

        glBindVertexArray(0)
        self._in_context = True

    def _remove_from_context(self):
        """Remove the object from the current OpenGL context.
        """
        if self._in_context:
            glDeleteVertexArrays(1, [self._vaid])
            glDeleteBuffers(len(self._buffers), self._buffers)
            self._in_context = False

    def _bind(self):
        """Bind this object in the current OpenGL context and load its shader.
        """
        if not self._in_context:
            raise ValueError('Cannot bind a SceneObject that has not been added to a context')

        # Bind Vertex Arrays
        glBindVertexArray(self._vaid)

    def _unbind(self):
        """Unbind this object in the current OpenGL context.
        """
        if not self._in_context:
            raise ValueError('Cannot unbind a SceneObject that has not been added to a context')

        glBindVertexArray(0)

    def _compute_transparency(self):
        # Check diffuse material color first
        if self.material.diffuse.ndim == 1 and self.material.diffuse.shape[0] == 4:
            if self.material.diffuse[3] < 1.0:
                return True
        elif self.material.diffuse.ndim == 3 and self.material.diffuse.shape[2] == 4:
            if np.any(self.material.diffuse[:,:,3] < 1.0):
                return True

        # Check vertex or face colors
        if self.vertex_colors is not None and self.vertex_colors.shape[1] == 4:
            if np.any(self.vertex_colors[:,3] < 1.0):
                return True
        if self.face_colors is not None and self.face_colors.shape[1] == 4:
            if np.any(self.face_colors[:,3] < 1.0):
                return True

        return False

    def _update_flags(self):
        """Compute the shading flags for this object.
        """
        self._vertex_buffer_flags = VertexBufferFlags.POSITION | VertexBufferFlags.NORMAL
        self._vertex_array_flags = VertexArrayFlags.TRIANGLES
        self._texture_flags = TextureFlags.NONE

        # Compute buffer flags
        if self.vertex_colors is not None or self.face_colors is not None:
            self._vertex_buffer_flags |= VertexBufferFlags.COLOR
        if self.texture_coords is not None:
            self._vertex_buffer_flags |= VertexBufferFlags.TEXTURE
        if self.material.normal is not None and self.material.normal.ndim > 1:
            self._vertex_buffer_flags |= VertexBufferFlags.TANGENT
            self._vertex_buffer_flags |= VertexBufferFlags.BITANGENT

        # Compute array flags
        if self.poses is not None:
            self._vertex_array_flags |= VertexArrayFlags.INSTANCED
        if self.material.smooth:
            self._vertex_array_flags |= VertexArrayFlags.ELEMENTS

        # Compute texture flags
        if self.material.diffuse.ndim > 1:
            self._texture_flags |= TextureFlags.DIFFUSE
        if self.material.specular.ndim > 1:
            self._texture_flags |= TextureFlags.SPECULAR
        if self.material.emission is not None and self.material.emission.ndim > 1:
            self._texture_flags |= TextureFlags.EMISSION
        if self.material.normal is not None:
            self._texture_flags |= TextureFlags.NORMAL
        if self.material.height is not None:
            self._texture_flags |= TextureFlags.HEIGHT


    @staticmethod
    def from_trimesh(mesh, material=None, texture_coords=None, is_visible=True, poses=None):
        """Create a SceneObject from a :obj:`trimesh.Trimesh`.

        Parameters
        ----------
        mesh : :obj:`trimesh.Trimesh`
            A triangular mesh.
        material : :obj:`Material`
            The material of the object. If not specified, a default grey material
            will be used.
        texture_coords : (n, 2) float, optional
            Texture coordinates for vertices, if needed.
        is_visible : bool
            If False, the object will not be rendered.
        poses : (n,4,4) float
            If specified, makes this object an instanced object.
            List of poses for each instance relative to object base frame.

        Returns
        -------
        scene_object : :obj:`SceneObject`
            The scene object created from the triangular mesh.
        """
        vertex_colors = None
        face_colors = None

        if mesh.visual.defined:
            if mesh.visual.kind == 'vertex':
                vertex_colors = mesh.visual.vertex_colors / 255.0
            elif mesh.visual.kind == 'face':
                face_colors = mesh.visual.face_colors / 255.0

        return MeshSceneObject(vertices=mesh.vertices, vertex_normals=mesh.vertex_normals,
                               faces=mesh.faces, face_normals=mesh.face_normals,
                               vertex_colors=vertex_colors, face_colors=face_colors,
                               material=material, texture_coords=texture_coords,
                               is_visible=is_visible, poses=poses)
