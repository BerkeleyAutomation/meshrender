Scene Objects
=============
Objects in a scene are represented by a triangluar mesh (a `Trimesh` from the `trimesh` package),
a 3D pose relative to the world (a `RigidTransform` from `autolab_core`), and a set of
material properties (`MaterialProperties`).

SceneObject
~~~~~~~~~~~
.. autoclass:: meshrender.SceneObject

InstancedSceneObject
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: meshrender.InstancedSceneObject
