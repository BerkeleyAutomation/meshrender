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
                 T_obj_world=None,
                 material=None,
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
        if T_obj_world is None:
            T_obj_world = RigidTransform(from_frame='obj', to_frame='world')
        if material is None:
            material = MaterialProperties()
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
    def __init__(self, mesh, poses=None, raw_pose_data=None, colors=None,
                 T_obj_world=None,
                 material=None,
                 enabled=True):
        """Initialize a scene object with the given mesh, pose, and material.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            A mesh representing the object's geometry.
        poses : list of autolab_core.RigidTransform
            A set of poses, one for each instance of the scene object,
            relative to the full object's origin.
        raw_pose_data : (4*n,4) float or None
            A numpy array containing raw pose data, where each row is a column of a point's
            homogenous transform matrix. If not present, poses must be present.
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
        self._raw_pose_data = raw_pose_data

        if self._raw_pose_data is None:
            if self._poses is None:
                raise ValueError('Either poses or raw_pose_data must be specified')
            self._raw_pose_data = np.zeros((4*len(self._poses), 4))
            for i, pose in enumerate(self._poses):
                self._raw_pose_data[i*4:(i+1)*4,:] = pose.matrix.T

        self._n_instances = self._raw_pose_data.shape[0] // 4

        self._colors = colors
        if self._colors is None:
            self._colors = np.tile(material.color, (self._n_instances,1))

    @property
    def poses(self):
        """list of autolab_core.RigidTransform: A set of poses for each instance relative to the object's origin.
        """
        return self._poses

    @property
    def raw_pose_data(self):
        """(4*n,4) float: Raw data for pose matrices.
        """
        return self._raw_pose_data

    @property
    def colors(self):
        """(n,3) float: The color of each instance.
        """
        return self._colors

    @property
    def n_instances(self):
        """int: The number of instances of this object.
        """
        return self._n_instances
