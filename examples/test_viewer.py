"""Examples of using meshrender for viewing and offscreen rendering.
"""
import os
import numpy as np
import trimesh

from meshrender import PerspectiveCamera,\
                       DirectionalLight, SpotLight,\
                       MetallicRoughnessMaterial,\
                       Primitive, Mesh, Node, Scene,\
                       Viewer, OffscreenRenderer

#==============================================================================
# Mesh creation
#==============================================================================

#------------------------------------------------------------------------------
# Creating textured meshes from trimeshes
#------------------------------------------------------------------------------

# Fuze trimesh
fuze_trimesh = trimesh.load('./models/fuze.obj', process=False)
fuze_mesh = Mesh.from_trimesh(fuze_trimesh)

# Drill trimesh
drill_trimesh = trimesh.load('./models/drill.obj', process=False)
drill_mesh = Mesh.from_trimesh(drill_trimesh)

# Water bottle trimesh
bottle_gltf = trimesh.load('./models/WaterBottle.glb', process=False)
bottle_trimesh = bottle_gltf.geometry[list(bottle_gltf.geometry.keys())[0]]
bottle_mesh = Mesh.from_trimesh(bottle_trimesh)

#------------------------------------------------------------------------------
# Creating meshes with per-vertex colors
#------------------------------------------------------------------------------
boxv_trimesh = trimesh.creation.boxv(extents=0.05*np.ones(3))
boxv_vertex_colors = np.random.uniform(size=(boxv_trimesh.vertices.shape))
boxv_trimesh.visual.vertex_colors = boxv_vertex_colors
boxv_mesh = Mesh.from_trimesh(boxv_trimesh)

#------------------------------------------------------------------------------
# Creating meshes with per-face colors
#------------------------------------------------------------------------------
boxf_trimesh = trimesh.creation.box(extents=0.05*np.ones(3))
boxf_face_colors = np.random.uniform(size=(boxf_trimesh.vertices.shape[0]*3,3))
boxf_trimesh.visual.face_colors = boxf_face_colors
boxf_mesh = Mesh.from_trimesh(boxf_trimesh)

#------------------------------------------------------------------------------
# Creating meshes from point clouds
#------------------------------------------------------------------------------
points = trimesh.creation.icosphere(radius=0.05).vertices
points_mesh = Mesh.from_points(points)

#==============================================================================
# Light creation
#==============================================================================

direc_l = DirectionalLight(color=np.ones(3), intensity=10.0)
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                   innerConeAngle=np.pi/16, outerConeAngle=np.pi/8)
point_l = PointLight(color=np.ones(3), intensity=10.0)

#==============================================================================
# Camera creation
#==============================================================================

cam = PerspectiveCamera(yfov=(np.pi / 2.0))

#==============================================================================
# Scene creation
#==============================================================================

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02]))

#==============================================================================
# Adding objects to the scene
#==============================================================================

#------------------------------------------------------------------------------
# By manually creating nodes
#------------------------------------------------------------------------------
fuze_node = Node(mesh=fuze_mesh, translation=np.array([0.1, 0.0, 0.0]))
scene.add_node(fuze_node)

#------------------------------------------------------------------------------
# By using the add() utility function
#------------------------------------------------------------------------------
scene.add(drill_mesh, pose=np.eye(4))
scene.add(cam)




scene.add(SpotLight(color=np.ones(3), intensity=10.0,

# Start with an empty scene
scene = Scene(ambient_light=np.ones(3)*0.02)#np.array([1.0, 0.0, 0.0]))

#====================================
# Add objects to the scene
#====================================

# Begin by loading meshes
#pawn_mesh = trimesh.load_mesh('./models/fuze.obj', process=False)
#s = trimesh.load('~/Downloads/WaterBottle.glb', process=False)
s = trimesh.load('~/Downloads/BoomBox.glb')
#s = trimesh.load('~/Downloads/ReciprocatingSaw.glb')
#s = trimesh.load('~/Downloads/Lantern.glb', process=False)
mesh_key = list(s.geometry.keys())[0]
pawn_mesh = s.geometry[mesh_key]
pawn_mesh.apply_scale(10.0)
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
#pawn_obj = Mesh.from_points(pawn_mesh.vertices, pawn_mesh.visual.to_color().vertex_colors)
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

lm2 = np.eye(4)
z = np.array([0.0, 1.0, 1.0])
z = z / np.linalg.norm(z)
x = np.array([0.0, -z[2], z[2]])
y = np.cross(z, x)
lm2[:3,:3] = np.c_[x,y,z]
#scene.add(DirectionalLight(color=np.ones(3), intensity=10.0), pose=lm)
lm2[:3,3] = np.array([0.0, 0.0, 0.1]) + z * 0.5

scene.add(SpotLight(color=np.ones(3), intensity=10.0, innerConeAngle=np.pi/16, outerConeAngle=np.pi/8), pose=lm)
#scene.add(SpotLight(color=np.ones(3), intensity=10.0, innerConeAngle=np.pi/16, outerConeAngle=np.pi/8), pose=lm2)
#scene.add(DirectionalLight(color=np.ones(3), intensity=10.0), pose=lm)
#scene.add(DirectionalLight(color=np.ones(3), intensity=10.0), pose=lm2)
#scene.add(PointLight(color=np.ones(3), intensity=1.0), pose=lm)
#scene.add(bar_obj, pose=pawn_pose.matrix)
cm = np.eye(4)
cm[:3,3] = np.array([0.0, 0.0, 0.25])
scene.add(PerspectiveCamera(yfov=np.pi/2.0), pose=cm)



r = OffscreenRenderer(640, 480)
color, depth = r.render(scene)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(color)
plt.show()

r.delete()
