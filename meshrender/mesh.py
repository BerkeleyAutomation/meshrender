import abc
import numpy as np
import six
import trimesh

from .primitive import Primitive
from .constants import GLTF
from .material import MetallicRoughnessMaterial

class Mesh(object):

    def __init__(self, primitives, name=None, weights=None, is_visible=True):
        self.primitives = primitives
        self.name = name
        self.weights = weights
        self.is_visible = is_visible

        self._bounds = None

    @property
    def bounds(self):
        if self._bounds is None:
            bounds = np.array([[np.infty, np.infty, np.infty],
                               [-np.infty, -np.infty, -np.infty]])
            for p in self.primitives:
                bounds[0] = np.minimum(bounds[0], p.bounds[0])
                bounds[1] = np.maximum(bounds[1], p.bounds[1])
            self._bounds = bounds
        return self._bounds

    @property
    def is_transparent(self):
        for p in self.primitives:
            if p.is_transparent:
                return True
        return False

    @staticmethod
    def from_points(points, colors=None, is_visible=True, poses=None):
        primitive = Primitive(
            positions=points,
            color_0=colors,
            mode=GLTF.POINTS,
            poses=poses
        )
        mesh = Mesh(primitives=[primitive], is_visible=is_visible)
        return mesh


    @staticmethod
    def from_trimesh(mesh, material=None, texcoords=None, is_visible=True, poses=None, smooth=True):
        """Create a Mesh from a :obj:`trimesh.Trimesh`.

        Parameters
        ----------
        mesh : :obj:`trimesh.Trimesh`
            A triangular mesh.
        material : :obj:`Material`
            The material of the object. If not specified, a default grey material
            will be used.
        texcoord_0 : (n, 2) float, optional
            Texture coordinates for positions, if needed.
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
        positions = None
        normals = None
        color_0 = None
        texcoord_0 = texcoords
        indices = None
        mode = GLTF.TRIANGLES

        def get_material(tm_material):
            material = None

            # Simple material
            if isinstance(tm_material. trimesh.visual.texture.SimpleMaterial):
                material = MetallicRoughnessMaterial(
                    alphaMode='BLEND',
                    baseColorTexture=tm_material.image
                )

            # PBRMaterial
            elif isinstance(tm_material, trimesh.visual.texture.PBRMaterial):
                material = MetallicRoughnessMaterial(
                    normalTexture=tm_material.normalTexture,
                    occlusionTexture=tm_material.occlusionTexture,
                    emissiveTexture=tm_material.emissiveTexture,
                    emissiveFactor=tm_material.emissiveFactor,
                    alphaMode='BLEND',
                    baseColorFactor=tm_material.baseColorFactor,
                    baseColorTexture=tm_material.baseColorTexture,
                    metallicFactor=tm_material.metallicFactor,
                    metallicRoughnessTexture=tm_material.metallicRoughnessTexture
                )
            return material

        default_material = MetallicRoughnessMaterial(
            alphaMode='BLEND', baseColorFactor=np.array([0.5, 0.5, 0.5, 1.0])
        )


        # Compute positions, normals, texture coordinates, colors, and indices
        if smooth:
            positions = mesh.vertices.copy()
            normals = mesh.vertex_normals.copy()
            indices = mesh.faces.copy()
            if mesh.visual.defined and material is None:
                if mesh.visual.kind == 'vertex':
                    color_0 = mesh.visual.vertex_colors.copy()
                    material = default_material
                elif mesh.visual.kind == 'face':
                    raise ValueError('Cannot use face colors with smooth mesh')
                elif mesh.visual.kind == 'texture':
                    texcoord_0 = mesh.visual.uv.copy()
                    material = get_material(mesh.visual.material)
        else:
            positions = mesh.vertices[mesh.faces]
            normals = np.repeat(mesh.face_normals, 3, axis=0)
            if mesh.visual.defined and material is None:
                if mesh.visual.kind == 'vertex':
                    color_0 = mesh.visual.vertex_colors[mesh.faces].\
                        reshape(3*len(mesh.faces), mesh.visual.vertex_colors.shape[1])
                    material = default_material
                elif mesh.visual.kind == 'face':
                    color_0 = np.repeat(mesh.face_colors, 3, axis=0)
                    material = default_material
                elif mesh.visual.kind == 'texture':
                    texcoord_0 = mesh.visual.uv[mesh.faces]
                    material = get_material(mesh.visual.material)

        primitive = Primitive(
            positions=positions,
            normals=normals,
            texcoord_0=texcoord_0,
            color_0=color_0,
            indices=indices,
            material=material,
            mode=mode,
            poses=poses
        )

        mesh = Mesh(primitives=[primitive], is_visible=is_visible)

        return mesh
