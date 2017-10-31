import numpy as np
import trimesh
from autolab_core import RigidTransform
from perception import CameraIntrinsics, RenderMode, ColorImage, DepthImage

from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera, DirectionalLight, SceneViewer

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
    color = np.array([0.1, 0.1, 0.1]),
    k_a = 0.3,
    k_d = 1.0,
    k_s = 0.5,
    alpha = 10.0,
    smooth=False
)
bar_material = MaterialProperties(
    color = np.array([0.1, 0.1, 0.5]),
    k_a = 0.3,
    k_d = 1.0,
    k_s = 0.5,
    alpha = 10.0,
    smooth=False
)

# Create SceneObjects for each object
pawn_obj = SceneObject(pawn_mesh, pawn_pose, pawn_material)
bar_obj = SceneObject(bar_mesh, bar_pose, bar_material)

# Add the SceneObjects to the scene
scene.add_object('pawn', pawn_obj)
#scene.add_object('bar', bar_obj)

#====================================
# Add lighting to the scene
#====================================

# Create an ambient light
ambient = AmbientLight(
    color=np.array([1.0, 1.0, 1.0]),
    strength=2.0
)

# Create a point light

points = []
for i in range(8):
    points.append(
        PointLight(
            location=np.array([-4.0+i, 4.0, 4.0]),
            color=np.array([1.0, 1.0, 1.0]),
            strength=4.0
        )
    )

direction = np.array([1.0, 1.0, -1.0])
direction = direction / np.linalg.norm(direction)

direc = DirectionalLight(
    direction=direction,
    color=np.array([1.0, 1.0, 1.0]),
    strength=1.0

)

# Add the lights to the scene
scene.ambient_light = ambient # only one ambient light per scene
for i, point in enumerate(points):
    scene.add_light('point_light_{}'.format(i), point)
scene.add_light('direc', direc)

#====================================
# Add a camera to the scene
#====================================

# Set up camera intrinsics
ci = CameraIntrinsics(
    frame = 'camera',
    fx = 20*525.0,
    fy = 20*525.0,
    cx = 320.0,
    cy = 240.0,
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
    translation = np.array([-4, 0.0, 0.0]),
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


wrapped_color.save('color.jpg')
wrapped_depth.save('depth.jpg')

v = SceneViewer(scene)

