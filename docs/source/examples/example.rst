Example Usage
=============

Everything in the `meshrender` model revolves around the `Scene` object,
which manages the lights, objects, and camera in the world.

Simple Example
--------------

In this example, we will render a pair of triangular meshes, illuminated by a point light source.

.. code-block:: python

    import numpy as np
    import trimesh
    from autolab_core import RigidTransform
    from perception import CameraIntrinsics, RenderMode

    from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera

    # Start with an empty scene
    scene = Scene()

    #====================================
    # Add objects to the scene
    #====================================

    # Begin by loading meshes
    cube_mesh = trimesh.load_mesh('cube.obj')
    sphere_mesh = trimesh.load_mesh('cube.obj')

    # Set up each object's pose in the world
    cube_pose = RigidTransform(
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
        from_frame='obj',
        to_frame='world'
    )
    sphere_pose = RigidTransform(
        rotation=np.eye(3),
        translation=np.array([1.0, 1.0, 0.0]),
        from_frame='obj',
        to_frame='world'
    )

    # Set up each object's material properties
    cube_material = MaterialProperties(
        color = np.array([0.1, 0.1, 0.5]),
        k_a = 0.3,
        k_d = 1.0,
        k_s = 1.0,
        alpha = 10.0,
        smooth=False
    )
    sphere_material = MaterialProperties(
        color = np.array([0.1, 0.1, 0.5]),
        k_a = 0.3,
        k_d = 1.0,
        k_s = 1.0,
        alpha = 10.0,
        smooth=True
    )

    # Create SceneObjects for each object
    cube_obj = SceneObject(cube_mesh, cube_pose, cube_material)
    sphere_obj = SceneObject(sphere_mesh, sphere_pose, sphere_material)

    # Add the SceneObjects to the scene
    scene.add_object('cube', cube_obj)
    scene.add_object('sphere', sphere_obj)

    #====================================
    # Add lighting to the scene
    #====================================

    # Create an ambient light
    ambient = AmbientLight(
        color=np.array([1.0, 1.0, 1.0]),
        strength=1.0
    )

    # Create a point light
    point = PointLight(
        position=np.array([1.0, 2.0, 3.0]),
        color=np.array([1.0, 1.0, 1.0]),
        strength=10.0
    )

    # Add the lights to the scene
    scene.ambient_light = ambient # only one ambient light per scene
    scene.add_light('point_light_one', point)

    #====================================
    # Add a camera to the scene
    #====================================

    # Set up camera intrinsics
    ci = CameraIntrinsics(
        frame = 'camera',
        fx = 525.0,
        fy = 525.0,
        cx = 319.5,
        cy = 239.5,
        skew=0.0,
        height=480,
        width=640
    )

    # Set up the camera pose (z axis faces away from scene, x to right, y up)
    cp = RigidTransform(
        rotation = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.0,  0.0],
            [1.0, 0.0,  0.0]
        ]),
        translation = np.array([-0.3, 0.0, 0.0]),
        from_frame='camera',
        to_frame='world'
    )

    # Create a VirtualCamera
    camera = VirtualCamera(ci, cp)

    # Add the camera to the scene
    scene.camera = camera

    #====================================
    # Render images
    #====================================

    # Render raw numpy arrays containing color and depth
    color_image_raw, depth_image_raw = scene.render(render_color=True)

    # Alternatively, just render a depth image
    depth_image_raw = scene.render(render_color=False)

    # Alternatively, collect wrapped images
    wrapped_color, wrapped_depth, wrapped_segmask = scene.wrapped_render(
        [RenderMode.COLOR, RenderMode.DEPTH, RenderMode.SEGMASK]
    )

