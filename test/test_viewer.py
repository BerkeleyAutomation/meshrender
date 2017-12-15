import numpy as np
import trimesh
from autolab_core import RigidTransform
from perception import CameraIntrinsics, RenderMode, ColorImage, DepthImage
import gc
import resource

from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera, DirectionalLight, SceneViewer, UniformPlanarWorksurfaceImageRandomVariable, InstancedSceneObject


def main():
    # Start with an empty scene
    scene = Scene()

    #====================================
    # Add objects to the scene
    #====================================

    # Begin by loading meshes
    pawn_mesh = trimesh.load_mesh('./models/pawn.obj')
    #pawn_mesh = trimesh.load_mesh('./models/pawn_large.obj')
    bar_mesh = trimesh.load_mesh('./models/bar_clamp.obj')

    # Set up each object's pose in the world
    pawn_pose = RigidTransform(
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
        from_frame='obj',
        to_frame='world'
    )
    bar_pose = RigidTransform(
        rotation=np.eye(3),
        translation=np.array([0.1, 0.07, 0.00]),
        from_frame='obj',
        to_frame='world'
    )

    # Set up each object's material properties
    pawn_material = MaterialProperties(
        color = 5.0*np.array([0.1, 0.1, 0.1]),
        k_a = 0.3,
        k_d = 0.5,
        k_s = 0.2,
        alpha = 10.0,
        smooth=False,
        wireframe=False
    )
    #bar_material = MaterialProperties(
    #    color = 7.0*np.array([0.1, 0.1, 0.1]),
    #    k_a = 0.5,
    #    k_d = 0.3,
    #    k_s = 0.1,
    #    alpha = 10.0,
    #    smooth=False
    #)
    bar_material = pawn_material

    # Create SceneObjects for each object
    pawn_obj = SceneObject(pawn_mesh, pawn_pose, pawn_material)
    bar_obj = SceneObject(bar_mesh, bar_pose, bar_material)
    pawn_inst_obj = InstancedSceneObject(pawn_mesh, [pawn_pose, bar_pose], material=pawn_material)

    # Add the SceneObjects to the scene
    scene.add_object('pawn', pawn_inst_obj)
    scene.add_object('bar', bar_obj)

    #====================================
    # Add lighting to the scene
    #====================================

    # Create an ambient light
    ambient = AmbientLight(
        color=np.array([1.0, 1.0, 1.0]),
        strength=1.0
    )

    # Create a point light

    points = []
    #for i in range(6):
    #    points.append(
    #        PointLight(
    #            location=np.array([-3.0, 3.0-i, 3.0]),
    #            color=np.array([1.0, 1.0, 1.0]),
    #            strength=4.0
    #        )
    #    )
    #
    #for i, point in enumerate(points):
    #    scene.add_light('point_{}'.format(i), point)

    # Add the lights to the scene
    scene.ambient_light = ambient # only one ambient light per scene

    dl = DirectionalLight(
        direction=np.array([0.0, 0.0, -1.0]),
        color=np.array([1.0, 1.0, 1.0]),
        strength=2.0
    )
    scene.add_light('direc', dl)

    #====================================
    # Add a camera to the scene
    #====================================

    # Set up camera intrinsics
    ci = CameraIntrinsics(
        frame = 'camera',
        fx = 525.0,
        fy = 525.0,
        cx = 320.0,
        cy = 240.0,
        skew=0.0,
        height=480,
        width=640
    )

    # Set up the camera pose (z axis faces away from scene, x to right, y up)
    cp = RigidTransform(
        rotation = np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0,  0.0],
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

    scene.close_renderer()

    wrapped_color.save('color.jpg')
    wrapped_depth.save('depth.jpg')

    # Test random variables
    cfg = {
    'min_f': 520,
    'max_f': 530,
    'min_cx': 299.5,
    'max_cx': 299.5,
    'min_cy': 299.5,
    'max_cy': 299.5,
    'im_width': 600,
    'im_height': 600,

    # Camera pose
    'min_radius': 0.5,
    'max_radius': 0.7,
    'min_elev': 0.1,
    'max_elev': 10.0, # in degrees
    'min_az': 0.0,
    'max_az': 360.0, # degrees
    'min_roll': -0.2,
    'max_roll': 0.2, # degrees

    # Object pose
    'min_x': -0.01,
    'max_x': 0.01,
    'min_y': -0.01,
    'max_y': 0.01
    }

    #urv = UniformPlanarWorksurfaceImageRandomVariable('pawn', scene, [RenderMode.COLOR], 'camera', cfg)
    #renders = urv.sample(10, front_and_back=True)

    #for i, render in enumerate(renders):
    #    color = render.renders[RenderMode.COLOR]
    #    color.save('random_{}.jpg'.format(i))

    #for i in range(2):
    #    v = SceneViewer(scene, raymond_lighting=True)
    #    gc.collect()

if __name__ == '__main__':
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE|gc.DEBUG_COLLECTABLE)
    main()
    print gc.garbage
    print '\n'
    for o in gc.garbage:
        print 'Object: {}'.format(o)
        print '-'*40
        for r in gc.get_referrers(o):
            print '{}'.format(r)
            print ''
    print "\n"*5

