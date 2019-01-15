import pyglet
pyglet.options['shadow_window'] = False

import numpy as np
import trimesh

import os
#os.environ['MESHRENDER_EGL_OFFSCREEN'] = 't'
from meshrender import Scene, Material, Mesh, SceneViewer, DirectionalLight, MetallicRoughnessMaterial, SpotLight, PointLight

# Start with an empty scene
scene = Scene()#np.array([1.0, 0.0, 0.0]))

#====================================
# Add objects to the scene
#====================================

# Begin by loading meshes
pawn_mesh = trimesh.load_mesh('./models/fuze.obj', process=False)
#s = trimesh.load('~/Downloads/WaterBottle.glb', process=False)
#s = trimesh.load('~/Downloads/BoomBox.glb')
#s = trimesh.load('~/Downloads/ReciprocatingSaw.glb')
#s = trimesh.load('~/Downloads/Lantern.glb', process=False)
#mesh_key = list(s.geometry.keys())[0]
#pawn_mesh = s.geometry[mesh_key]
pawn_pose = np.eye(4)
#pawn_mesh = trimesh.creation.icosahedron()
#colors = (255*np.random.uniform(size=pawn_mesh.vertices.shape)).astype(np.uint8)
#pawn_mesh.visual.vertex_colors = colors
#pawn_mesh = trimesh.load_mesh('./models/pawn_large.obj')

# Set up each object's pose in the world

## Set up each object's material properties
#pawn_material = Material(
#    diffuse = np.array([0.5, 0.5, 0.0]),#0.5*np.ones(3),
#    specular = 0.3*np.ones(3),
#    shininess = 10.0,
#    smooth=True,
#    wireframe=False
#)
#
##bar_material = MaterialProperties(
##    color = 7.0*np.array([0.1, 0.1, 0.1]),
##    k_a = 0.5,
##    k_d = 0.3,
##    k_s = 0.1,
##    alpha = 10.0,
##    smooth=False
##)
#bar_material = pawn_material.copy()
#bar_material.diffuse = np.array([1.0, 0.0, 0.0])

# Create SceneObjects for each object
pawn_obj = Mesh.from_trimesh(pawn_mesh)
#Jbar_obj = MeshSceneObject.from_trimesh(bar_mesh, bar_material)

# Add the SceneObjects to the scene
scene.add(pawn_obj, pose=pawn_pose)

# PLANE
vertices = np.array([
    [-0.2, -0.2, 0.0],
    [-0.2, 0.2, 0.0],
    [0.2, 0.2, 0.0],
    [0.2, -0.2, 0.0],
])
faces = np.array([
    [0,2,1],
    [0,3,2]
])
texture_coords = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0, 0.0]
])
vertex_colors = np.array([
    [255,0,0,255],
    [255,0,0,255],
    [255,0,0,255],
    [255,0,0,255],
])
x = trimesh.Trimesh(vertices, faces, texture_coords)#, vertex_colors=vertex_colors)
pobj = Mesh.from_trimesh(x, material=MetallicRoughnessMaterial(baseColorFactor=np.array([1.0, 0.0, 0.0, 1.0])))
scene.add(pobj)

lm = np.eye(4)
z = np.array([1.0, 0.0, 1.0])
z = z / np.linalg.norm(z)
x = np.array([-z[0], 0.0, z[0]])
y = np.cross(z, x)
lm[:3,:3] = np.c_[x,y,z]
#scene.add(DirectionalLight(color=np.ones(3), intensity=10.0), pose=lm)
lm[:3,3] = np.array([0.0, 0.0, 0.1]) + z * 0.5
scene.add(SpotLight(color=np.ones(3), intensity=10.0, innerConeAngle=np.pi/16, outerConeAngle=np.pi/8), pose=lm)
#scene.add(PointLight(color=np.ones(3), intensity=1.0), pose=lm)
#scene.add(bar_obj, pose=pawn_pose.matrix)

#====================================
# Add lighting to the scene
#====================================

## Create an ambient light
#ambient = AmbientLight(
#    color=np.array([1.0, 1.0, 1.0]),
#    strength=1.0
#)
#
## Create a point light
#
#points = []
##for i in range(6):
##    points.append(
##        PointLight(
##            location=np.array([-3.0, 3.0-i, 3.0]),
##            color=np.array([1.0, 1.0, 1.0]),
##            strength=4.0
##        )
##    )
##
##for i, point in enumerate(points):
##    scene.add_light('point_{}'.format(i), point)
#
## Add the lights to the scene
#scene.ambient_light = ambient # only one ambient light per scene
#
#dl = DirectionalLight(
#    direction=np.array([0.0, 0.0, -1.0]),
#    color=np.array([1.0, 1.0, 1.0]),
#    strength=2.0
#)
#scene.add_light('direc', dl)

#====================================
# Add a camera to the scene
#====================================

## Set up camera intrinsics
#ci = CameraIntrinsics(
#    frame = 'camera',
#    fx = 525.0,
#    fy = 525.0,
#    cx = 320.0,
#    cy = 240.0,
#    skew=0.0,
#    height=480,
#    width=640
#)
#
## Set up the camera pose (z axis faces away from scene, x to right, y up)
#cp = RigidTransform(
#    rotation = np.array([
#        [0.0, 0.0, 1.0],
#        [0.0, -1.0,  0.0],
#        [1.0, 0.0,  0.0]
#    ]),
#    translation = np.array([-0.3, 0.0, 0.0]),
#    from_frame='camera',
#    to_frame='world'
#)
#
## Create a VirtualCamera
#camera = VirtualCamera(ci, cp)
#
## Add the camera to the scene
#scene.camera = camera

#====================================
# Render images
#====================================

## Render raw numpy arrays containing color and depth
#color_image_raw, depth_image_raw = scene.render(render_color=True)
#
## Alternatively, just render a depth image
#depth_image_raw = scene.render(render_color=False)
#
## Alternatively, collect wrapped images
#wrapped_color, wrapped_depth, wrapped_segmask = scene.wrapped_render(
#    [RenderMode.COLOR, RenderMode.DEPTH, RenderMode.SEGMASK]
#)
#
#wrapped_color.save('output/color.jpg')
#wrapped_depth.save('output/depth.jpg')
#
## Test random variables
#cfg = {
#    'focal_length': {
#        'min' : 520,
#        'max' : 530,
#    },
#    'delta_optical_center': {
#        'min' : 0.0,
#        'max' : 0.0,
#    },
#    'radius': {
#        'min' : 0.5,
#        'max' : 0.7,
#    },
#    'azimuth': {
#        'min' : 0.0,
#        'max' : 360.0,
#    },
#    'elevation': {
#        'min' : 0.10,
#        'max' : 10.0,
#    },
#    'roll': {
#        'min' : -0.2,
#        'max' : 0.2,
#    },
#    'x': {
#        'min' : -0.01,
#        'max' : 0.01,
#    },
#    'y': {
#        'min' : -0.01,
#        'max' : 0.01,
#    },
#    'im_width': 600,
#    'im_height': 600
#}
#
#urv = UniformPlanarWorksurfaceImageRandomVariable('pawn', scene, [RenderMode.COLOR], 'camera', cfg)
#renders = urv.sample(10, front_and_back=True)
#
#for i, render in enumerate(renders):
#    color = render.renders[RenderMode.COLOR]
#    color.save('output/random_{}.jpg'.format(i))

v = SceneViewer(scene, raymond_lighting=True)
