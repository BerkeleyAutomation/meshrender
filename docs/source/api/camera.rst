Camera
======
The `VirualCamera` object provides a wrapper for the intrinsics and pose of
a camera in a scene.
The camera's intrinsics are represented by a `CameraIntrinsics` object from
the Berkeley AUTOLab's `perception` package, and the camera's pose is represented
by a `RigidTransform` object from the `autolab_core` package.

The camera's frame of reference is given by an x-axis pointing to the right,
a y-axis pointing up, and a z-axis pointing away from the scene (i.e. into the eye of the camera)
along the optical axis.

VirtualCamera
~~~~~~~~~~~~~
.. autoclass:: meshrender.VirtualCamera

