import abc
import numpy as np
import six

from .material import Material

from .constants import Shading

@six.add_metaclass(abc.ABCMeta)
class SceneObject(object):
    """An object in a scene.

    Attributes
    ----------
    vertices : (n,3) float
        Object vertices.
    """
    def __init__(self):
        pass

    @property
    def transparent(self):
        pass

    @property
    def shading_mode(self):
        pass

class PointCloudSceneObject(SceneObject):
    """A cloud of points.

    Attributes
    ----------
    vertices : (n,3) float
        Object vertices.
    vertex_colors : (n,3) or (n,4) float
        Colors of each vertex.
    """

    def __init__(self, vertices, vertex_colors=None):
        self.vertices = vertices
        self.vertex_colors = vertex_colors

        if self.vertex_colors is None:
            self.vertex_colors = 0.5 * np.ones(size=self.vertices.size)

    @property
    def transparent(self):
        if self.vertex_colors.shape[1] == 4:
            if np.any(self.vertex_colors[:,3] < 1.0):
                return True
        return False

    @property
    def shading_mode(self):
        return Shading.POINT_CLOUD

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
        Face normals, optionally specified.
    vertex_colors : (n,3) or (n,4), float
        Colors for each vertex, if desired. This is overridden by any texture maps.
    face_colors : (m,3) or (m,4), float
        Colors for each face, if desired. This is overridden by any texture maps or vertex colors.
    material : :obj:`Material`
        The material of the object. If not specified, a default grey material
        will be used.
    texture_coords : (n, 2) float, optional
        Texture coordinates for vertices, if needed.
    visible : bool
        If False, the object will not be rendered.
    """

    def __init__(self, vertices, vertex_normals=None, faces=None, face_normals=None,
                 vertex_colors=None, face_colors=None, material=None, texture_coords=None, visible=True):
        self.vertices = vertices
        self.vertex_normals = vertex_normals
        self.faces = faces
        self.face_normals = face_normals
        self.vertex_colors = vertex_colors
        self.face_colors = face_colors
        self.material = material
        self.texture_coords = texture_coords
        self.visible = visible

        if texture_coords is not None:
            if not texture_coords.shape == (self.vertices.shape[0], 2):
                raise ValueError('Incorrect texture coordinate shape: {}'.format(texture_coords.shape))

        if vertex_colors is not None:
            if not vertex_colors.shape[0] == self.vertices.shape[0]:
                raise ValueError('Incorrect vertex colors shape: {}'.format(vertex_colors.shape))

        if face_colors is not None:
            if not face_colors.shape[0] == self.face_colors.shape[0]:
                raise ValueError('Incorrect vertex colors shape: {}'.format(vertex_colors.shape))

        if self.material is None:
            self.material = Material(
                diffuse=np.array([0.5, 0.5, 0.5]),
                specular=np.array([0.1, 0.1, 0.1]),
                shininess=10.0,
                smooth=False,
                wireframe=False,
            )

    @property
    def transparent(self):
        shading_mode = self.shading_mode

        if shading_mode & (Shading.TEX_DIFF | Shading.TEX_SPEC):
            if shading_mode & Shading.TEX_DIFF:
                if self.material.diffuse.ndim == 3 and self.material.diffuse.shape[2] == 4:
                    if np.any(self.material.diffuse[:,:,3]) < 1.0:
                        return True
            if shading_mode & Shading.TEX_SPEC:
                if self.material.specular.ndim == 3 and self.material.specular.shape[2] == 4:
                    if np.any(self.material.specular[:,:,3]) < 1.0:
                        return True
        elif shading_mode & Shading.VERT_COLORS:
            if self.vertex_colors.shape[1] == 4:
                if np.any(self.vertex_colors[:,3] < 1.0):
                    return True
        elif shading_mode & Shading.FACE_COLORS:
            if self.face_colors.shape[1] == 4:
                if np.any(self.face_colors[:,3] < 1.0):
                    return True
        else:
            if len(self.material.diffuse) == 4 and self.material.diffuse[3] < 1.0:
                return True
            if len(self.material.specular) == 4 and self.material.specular[3] < 1.0:
                return True
        return False

    @property
    def shading_mode(self):
        mode = Shading.DEFAULT
        if self.texture_coords is not None:
            if self.material.diffuse.ndim > 1:
                mode |= Shading.TEX_DIFF
            if self.material.specular.ndim > 1:
                mode |= Shading.TEX_SPEC
            if self.material.normal is not None and self.material.normal.ndim > 1:
                mode |= Shading.TEX_NORM
            if self.material.emission is not None and self.material.emission.ndim > 1:
                mode |= Shading.TEX_EMIT
        if mode == Shading.DEFUALT:
            if self.vertex_colors is not None:
                mode |= Shading.VERT_COLORS
            elif self.face_colors is not None:
                mode |= Shading.FACE_COLORS

    @staticmethod
    def from_trimesh(mesh, material=None, texture_coords=None, visible=True):
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
        visible : bool
            If False, the object will not be rendered.

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

        return SceneObject(vertices=mesh.vertices, vertex_normals=mesh.vertex_normals,
                           faces=mesh.faces, face_normals=mesh.face_normals,
                           vertex_colors=vertex_colors, face_colors=face_colors, material=material,
                           texture_coords=texture_coords, visible=visible)

class InstancedSceneObject(SceneObject):
    """An instanced triangular mesh scene object.

    Attributes
    ----------
    scene_object : :obj:`SceneObject`
        Base scene object to instance.
    poses : (n,4,4) float
        List of poses for each instance relative to base frame.
    """

    def __init__(self, scene_object, poses):
        self.scene_object = scene_object
        self.poses = poses

    @property
    def transparent(self):
        return self.scene_object.transparent

    @property
    def shading_mode(self):
        return self.scene_object.shading_mode | Shading.INSTANCED
