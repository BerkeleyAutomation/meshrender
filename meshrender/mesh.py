import abc
import numpy as np
import six
import trimesh

from OpenGL.GL import *

from .material import Material
from .constants import FLOAT_SZ, UINT_SZ, BufFlags, GeomFlags, TexFlags

class Mesh(object):

    def __init__(self, vertices, vertex_normals=None, faces=None, face_normals=None,
                 texture_coords=None, vertex_colors=None, face_colors=None,
                 material=None, is_visible=True, poses=None):

        self.vertices = vertices
        self.vertex_normals = vertex_normals
        self.faces = faces
        self.face_normals = face_normals
        self.texture_coords = texture_coords
        self.vertex_colors = vertex_colors
        self.face_colors = face_colors
        self.material = material
        self.is_visible = is_visible
        self.poses = poses

        self._bounds = None
        self._vaid = None
        self._buffers = []
        self._buf_flags = None
        self._geom_flags = None
        self._tex_flags = None

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = np.ascontiguousarray(value.astype(np.float32))
        self._bounds = None

    @property
    def vertex_normals(self):
        return self._vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.float32))
            if value.shape != self.vertices.shape:
                raise ValueError('Incorrect vertex normal shape')
        self._vertex_normals = value

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.uint32))
        self._faces = value

    @property
    def face_normals(self):
        return self._face_normals

    @face_normals.setter
    def face_normals(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.float32))
            if self.faces is None:
                raise ValueError('Cannot define face normals without faces')
            if value.shape != self.faces.shape:
                raise ValueError('Incorrect face normal shape')
        self._face_normals = value

    @property
    def texture_coords(self):
        return self._texture_coords

    @texture_coords.setter
    def texture_coords(self, value):
        if value is not None:
            value = np.ascontiguousarray(value.astype(np.float32))
            if value.ndim != 2 or value.shape[0] != self.vertices.shape[0] or value.shape[1] != 2:
                raise ValueError('Incorrect texture coordinate shape')
        self._texture_coords = value

    @property
    def vertex_colors(self):
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, value):
        if value is not None:
            value = np.ascontiguousarray(format_color_array(value, 4))
            if value.shape[0] != self.vertices.shape[0]:
                raise ValueError('Incorrect vertex color shape')
        self._vertex_colors = value

    @property
    def face_colors(self):
        return self._face_colors

    @face_colors.setter
    def face_colors(self, value):
        if value is not None:
            value = np.ascontiguousarray(format_color_array(value, 4))
        self._face_colors = value

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        # Create default material
        if value is None:
            smooth = (self.face_normals is not None)
            value = MetallicRoughnessMaterial(smooth=smooth, base_color_factor=np.ones(4))
        else:
            if not isinstance(value, Material):
                raise ValueError('Object material must be of type Material')
        self._material = value

    @property
    def poses(self):
        return self._poses

    @poses.setter
    def poses(self, value):
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
    def geom_flags(self):
        if self._geom_flags is None:
            self._geom_flags = self._compute_geom_flags()
        return self._geom_flags

    @property
    def tex_flags(self):
        if self._tex_flags is None:
            self._tex_flags = self._compute_tex_flags()
        return self._tex_flags

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

        # Vertices
        vertex_data = self.vertices
        attr_sizes = [3]

        # Normals
        normals = None
        if self.material.smooth and self.vertex_normals is not None:
            normals = self.vertex_normals
        elif self.vertex_normals is None:
            if self.face_normals is not None:
                normals = np.repeat(self.face_normals, 3, axis=0)
            elif self.faces is not None:
                triangles = self.vertices[self.faces]
                normals = trimesh.triangles.normals(triangles)
        if normals is not None:
            vertex_data = np.hstack((vertex_data, normals))
            attr_sizes.append(3)

        # Colors
        colors = None
        if self.vertex_colors is not None:
            if self.material.smooth:
                colors = self.vertex_colors
            elif faces is not None:
                colors = self.vertex_colors[self.faces].\
                              reshape(3*len(self.faces), self.vertex_colors.shape[1])
            else:
                colors = self.vertex_colors
        elif self.face_colors is not None:
            if self.material.smooth:
                raise ValueError('Cannot have face colors for smooth mesh')
            else:
                colors = np.repeat(self.face_colors, 3, axis=0)
        if colors is not None:
            vertex_data = np.hstack((vertex_data, colors))
            attr_sizes.append(4)

        # Texture Coordinates
        if self.texture_coords is not None:
            vertex_data = np.hstack((vertex_data, texture_coords))
            attr_sizes.append(2)

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
        self._remove_from_context()

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

    def _compute_transparency(self):
        """Compute whether or not this object is transparent.
        """
        if self.vertex_colors is not None:
            if np.any(self._vertex_colors[:,3] != 1.0):
                return True

        if self.face_colors is not None:
            if np.any(self._face_colors[:,3] != 1.0):
                return True

        if self.material.is_transparent:
            return True

        return False

    def _compute_buf_flags(self):
        buf_flags = BufFlags.POSITION

        if self.vertex_normals is not None or self.face_normals is not None or self.faces is not None:
            buf_flags |= BufFlags.NORMAL
        if self.material.requires_tangents:
            buf_flags |= BufFlags.TANGENT
        if self.texture_coords is not None:
            buf_flags |= BufFlags.TEXCOORD_0
        if self.vertex_colors is not None or self.face_colors is not None:
            buf_flags |= BufFlags.COLOR_0

        return buf_flags

    def _compute_geom_flags(self):
        geom_flags = GeomFlags.NONE

        if self.faces is not None:
            if self.material.smooth:
                geom_flags = GeomFlags.ELEMENTS
            else:
                geom_flags = GeomFlags.TRIANGLES
        else:
            geom_flags = GeomFlags.POINTS

        return geom_flags

    def _compute_tex_flags(self):
        tex_flags = self.material.tex_flags
        return tex_flags

    @staticmethod
    def from_trimesh(mesh, material=None, texture_coords=None, is_visible=True, poses=None, smooth=True):
        """Create a Mesh from a :obj:`trimesh.Trimesh`.

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
        scene_object : :obj:`Mesh`
            The scene object created from the triangular mesh.
        """
        vertex_colors = None
        face_colors = None

        # If we didn't provide a material, try to use the mesh's
        if mesh.visual.defined and material is None:
            if mesh.visual.kind == 'vertex':
                vertex_colors = mesh.visual.vertex_colors
            elif mesh.visual.kind == 'face':
                face_colors = mesh.visual.face_colors
            elif mesh.visual.kind == 'texture':
                texture_coords = np.array(mesh.visual.uv)

                tm_material = mesh.visual.material

                # Simple material
                if isinstance(tm_material. trimesh.visual.texture.SimpleMaterial):
                    material = MetallicRoughnessMaterial(
                        alpha_mode='BLEND',
                        smooth=smooth,
                        base_color_texture=tm_material.image
                    )

                # PBRMaterial
                elif isinstance(tm_material, trimesh.visual.texture.PBRMaterial):
                    material = MetallicRoughnessMaterial(
                        normal_texture=tm_material.normalTexture,
                        occlusion_texture=tm_material.occlusionTexture,
                        emissive_texture=tm_material.emissiveTexture,
                        emissive_factor=tm_material.emissiveFactor,
                        alpha_mode='BLEND',
                        smooth=smooth,
                        base_color_factor=tm_material.baseColorFactor,
                        base_color_texture=tm_material.baseColorTexture,
                        metallic_factor=tm_material.metallicFactor,
                        metallic_roughness_texture=tm_material.metallicRoughnessTexture
                    )

        return Mesh(vertices=mesh.vertices, vertex_normals=mesh.vertex_normals,
                    faces=mesh.faces, face_normals=mesh.face_normals,
                    vertex_colors=vertex_colors, face_colors=face_colors,
                    material=material, texture_coords=texture_coords,
                    is_visible=is_visible, poses=poses)
