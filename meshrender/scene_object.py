import numpy as np

from trimesh import Trimesh
from autolab_core import RigidTransform

from .material import MaterialProperties

class SceneObject(object):
    """A complete description of an object in a Scene.

    This includes its geometry (represented as a Trimesh), its pose in the world,
    and its material properties.
    """

    def __init__(self, mesh,
                 T_obj_world=RigidTransform(from_frame='obj', to_frame='world'),
                 material=MaterialProperties(),
                 enabled=True):
        """Initialize a scene object with the given mesh, pose, and material.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            A mesh representing the object's geometry.
        T_obj_world : autolab_core.RigidTransform
            A rigid transformation from the object's frame to the world frame.
        material : MaterialProperties
            A set of material properties for the object.
        enabled : bool
            If False, the object will not be rendered.
        """
        if not isinstance(mesh, Trimesh):
            raise ValueError('mesh must be an object of type Trimesh')
        if not isinstance(material, MaterialProperties):
            raise ValueError('material must be an object of type MaterialProperties')

        if material.smooth:
            mesh = mesh.smoothed()

        self._mesh = mesh
        self._material = material
        self.T_obj_world = T_obj_world
        self._enabled = True

    @property
    def enabled(self):
        """bool: If False, the object will not be rendered.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        self._enabled = enabled

    @property
    def mesh(self):
        """trimesh.Trimesh: A mesh representing the object's geometry.
        """
        return self._mesh

    @property
    def material(self):
        """MaterialProperties: A set of material properties for the object.
        """
        return self._material

    @property
    def T_obj_world(self):
        """autolab_core.RigidTransform: A rigid transformation from the object's frame to the world frame.
        """
        return self._T_obj_world

    @T_obj_world.setter
    def T_obj_world(self, T):
        if not isinstance(T, RigidTransform):
            raise ValueError('transform must be an object of type RigidTransform')
        self._T_obj_world = T

class InstancedSceneObject(SceneObject):
    """A scene object which consists as a set of identical objects.
    """
    def __init__(self, mesh, poses, colors=None,
                 T_obj_world=RigidTransform(from_frame='obj', to_frame='world'),
                 material=MaterialProperties(),
                 enabled=True):
        """Initialize a scene object with the given mesh, pose, and material.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            A mesh representing the object's geometry.
        poses : list of autolab_core.RigidTransform
            A set of poses, one for each instance of the scene object,
            relative to the full object's origin.
        colors : (n,3) float or None
            A set of colors for each instanced object. If None, the color specified in material
            properties is used for all instances.
        T_obj_world : autolab_core.RigidTransform
            A rigid transformation from the object's frame to the world frame.
        material : MaterialProperties
            A set of material properties for the object.
        enabled : bool
            If False, the object will not be rendered.
        """

        super(InstancedSceneObject, self).__init__(mesh, T_obj_world, material, enabled)
        self._poses = poses
        self._colors = colors
        if self._colors is None:
            self._colors = np.tile(material.color, (len(poses),1))

    @property
    def poses(self):
        """list of autolab_core.RigidTransform: A set of poses for each instance relative to the object's origin.
        """
        return self._poses

    @property
    def colors(self):
        """(n,3) float: The color of each instance.
        """
        return self._colors
