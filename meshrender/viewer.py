import os
import copy
try:
    from Tkinter import Tk, tkFileDialog as filedialog
except ImportError:
    try:
        from tkinter import Tk, filedialog as filedialog
    except:
        pass

import pyglet
pyglet.options['shadow_window'] = False
import pyglet.gl as gl
from pyglet import clock

import numpy as np
import imageio

_USE_EGL_OFFSCREEN = False
if 'MESHRENDER_EGL_OFFSCREEN' in os.environ:
    _USE_EGL_OFFSCREEN = True

import OpenGL

from .constants import OPEN_GL_MAJOR, OPEN_GL_MINOR, RenderFlags
from .light import DirectionalLight
from .node import Node
from .camera import PerspectiveCamera
from .trackball import Trackball
from .renderer import Renderer

class SceneViewer(pyglet.window.Window):
    """An interactive viewer for a 3D scene.

    This doesn't use the scene's camera - instead, it uses one based on a trackball.

    The basic commands for moving about the scene are given as follows:

    * To rotate the camera about the center of the scene, hold the left mouse button and drag the cursor.
    * To rotate the camera about its viewing axis, hold CTRL and then hold the left mouse button and drag the cursor.
    * To pan the camera, do one of the following:

        * Hold SHIFT, then hold the left mouse button and drag the cursor.
        * Hold the middle mouse button and drag the cursor.

    * To zoom the camera in or our, do one of the following:

        * Scroll the mouse wheel.
        * Hold the right mouse button and drag the cursor.

    Other keyboard commands are as follows:

    * z -- resets the view to the original view.
    * w -- toggles wireframe mode for each mesh in the scene.
    * a -- toggles rotational animation.
    * l -- toggles two-sided lighting
    * q -- quits the viewer
    * s -- saves the current image
    * r -- starts a recording session, pressing again stops (saves animation as .gif)
    """

    def __init__(self, scene, viewport_size=None,
                 render_flags=None, viewer_flags=None,
                 registered_keys=None, **kwargs):

        ########################################################################
        # Save attributes and flags
        ########################################################################
        if viewport_size is None:
            viewport_size = (640, 480)
        self._scene = scene
        self._viewport_size = viewport_size

        self._default_render_flags = {
            'flip_wireframe' : False,
            'all_wireframe' : False,
            'all_solid' : False,
            'shadows' : False,
            'vertex_normals' : False,
            'face_normals' : False,
            'cull_faces' : True,
        }
        self._default_viewer_flags = {
            'mouse_pressed' : False,
            'rotate' : False,
            'rotate_rate' : np.pi / 3.0,
            'rotate_axis' : np.array([0.0, 0.0, 1.0]),
            'central_node' : None,
            'record' : False,
            'use_raymond_lighting' : False,
            'use_direct_lighting' : False,
            'save_directory' : None,
            'window_title' : 'Scene Viewer',
            'refresh_rate': 30.0
        }

        self.render_flags = self._default_render_flags.copy()
        self.viewer_flags = self._default_viewer_flags.copy()
        self.viewer_flags['rotate_axis'] = self._default_viewer_flags['rotate_axis'].copy()
        if render_flags is not None:
            self.render_flags.update(render_flags)
        if viewer_flags is not None:
            self.viewer_flags.update(viewer_flags)
        for key in kwargs:
            if key in self.render_flags:
                self.render_flags[key] = kwargs[key]
            elif key in self.viewer_flags:
                self.viewer_flags[key] = kwargs[key]

        self._registered_keys = {}
        if registered_keys is not None:
            self._registered_keys = {
                ord(k.lower()) : registered_keys[k] for k in registered_keys
            }

        ########################################################################
        # Set up camera node
        ########################################################################
        self._camera_node = None
        self._prior_main_camera_node = None
        self._default_camera_pose = None
        self._trackball = None
        self._saved_frames = []

        # Extract main camera from scene and set up our mirrored copy
        if scene.main_camera_node is not None:
            n = scene.main_camera_node
            c = n.camera
            self._default_camera_pose = scene.get_pose(scene.main_camera_node)
            self._camera_node = Node(
                name='__viewer_camera__',
                matrix=self._default_camera_pose,
                camera=copy.copy(n.camera)
            )
            scene.add_node(self._camera_node)
            scene.main_camera_node = self._camera_node
            self._prior_main_camera_node = n
        else:
            self._default_camera_pose = self._compute_initial_camera_pose()
            self._camera_node = Node(
                name='__viewer_camera__',
                matrix=self._default_camera_pose,
                camera = PerspectiveCamera(yfov=np.pi / 6.0),
            )
            scene.add_node(self._camera_node)
            scene.main_camera_node = self._camera_node
        self._reset_view()

        # Set up raymond lights and direct lights
        self._raymond_lights = self._create_raymond_lights()
        self._direct_light = self._create_direct_light()

        ########################################################################
        # Initialize OpenGL context and renderer
        ########################################################################
        try:
            conf = gl.Config(sample_buffers=1, samples=4,
                             depth_size=24, double_buffer=True,
                             major_version=OPEN_GL_MAJOR,
                             minor_version=OPEN_GL_MINOR)
            super(SceneViewer, self).__init__(config=conf, resizable=True,
                                              width=self._viewport_size[0],
                                              height=self._viewport_size[1])
        except Exception as e:
            raise ValueError('Failed to initialize Pyglet window with an OpenGL 3+ context. ' \
                             'If you\'re logged in via SSH, ensure that you\'re running your script ' \
                             'with vglrun (i.e. VirtualGL). Otherwise, the internal error message was: ' \
                             '"{}"'.format(e.message))

        self._is_high_dpi = False
        if hasattr(self.context, '_nscontext'):
            self._is_high_dpi = True

        if self._is_high_dpi:
            self._renderer = Renderer(2 * self._viewport_size[0], 2*self._viewport_size[1])
        else:
            self._renderer = Renderer(self._viewport_size[0], self._viewport_size[1])

        # Start timing event
        clock.set_fps_limit(self.viewer_flags['refresh_rate'])
        clock.schedule_interval(SceneViewer.time_event, 1.0/self.viewer_flags['refresh_rate'], self)
        self.set_caption(self.viewer_flags['window_title'])

        # Start the event loop
        pyglet.app.run()

    @property
    def scene(self):
        return self._scene

    @property
    def viewport_size(self):
        return self._viewport_size

    def on_close(self):
        """Exit the event loop when the window is closed.
        """
        # Remove our camera and restore the prior one
        if self._camera_node is not None:
            self.scene.remove_node(self._camera_node)
        if self._prior_main_camera_node is not None:
            self.scene.main_camera_node = self._prior_main_camera_node

        # Delete renderer and clean up OpenGL context data
        self._renderer.delete()
        OpenGL.contextdata.cleanupContext()

        pyglet.app.exit()

    def on_draw(self):
        """Redraw the scene into the viewing window.
        """
        self._render()

    def on_resize(self, width, height):
        """Resize the camera and trackball when the window is resized.
        """
        self._viewport_size = (width, height)
        self._trackball.resize(self._viewport_size)
        if self._is_high_dpi:
            self._renderer.viewport_width = 2 * self._viewport_size[0]
            self._renderer.viewport_height = 2 * self._viewport_size[1]
        else:
            self._renderer.viewport_width = self._viewport_size[0]
            self._renderer.viewport_height = self._viewport_size[1]
        self.on_draw()

    def on_mouse_press(self, x, y, buttons, modifiers):
        """Record an initial mouse press.
        """
        self._trackball.set_state(Trackball.STATE_ROTATE)
        if (buttons == pyglet.window.mouse.LEFT):
            ctrl = (modifiers & pyglet.window.key.MOD_CTRL)
            shift = (modifiers & pyglet.window.key.MOD_SHIFT)
            if (ctrl and shift):
                self._trackball.set_state(Trackball.STATE_ZOOM)
            elif ctrl:
                self._trackball.set_state(Trackball.STATE_ROLL)
            elif shift:
                self._trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == pyglet.window.mouse.MIDDLE):
            self._trackball.set_state(Trackball.STATE_PAN)
        elif (buttons == pyglet.window.mouse.RIGHT):
            self._trackball.set_state(Trackball.STATE_ZOOM)

        self._trackball.down(np.array([x, y]))

        # Stop animating while using the mouse
        self.viewer_flags['mouse_pressed'] = True


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Record a mouse drag.
        """
        self._trackball.drag(np.array([x, y]))

    def on_mouse_release(self, x, y, button, modifiers):
        """Record a mouse release.
        """
        self.viewer_flags['mouse_pressed'] = False

    def on_mouse_scroll(self, x, y, dx, dy):
        """Record a mouse scroll.
        """
        self._trackball.scroll(dy)

    def on_key_press(self, symbol, modifiers):
        """Record a key press.
        """
        # First, check for registered key callbacks
        if symbol in self._registered_keys:
            tup = self._registered_keys[symbol]
            callback = None
            args = []
            kwargs = {}
            if not isinstance(tup, list) or isinstance(tup, tuple):
                callback = tup
            else:
                callback = tup[0]
                if len(tup) == 2:
                    args = tup[1]
                if len(tup) == 3:
                    kwargs = tup[2]
            callback(self, *args, **kwargs)
            return

        # Otherwise, use default key functions

        # W toggles through wireframe modes
        if symbol == pyglet.window.key.W:
            if self.render_flags['flip_wireframe']:
                self.render_flags['flip_wireframe'] = False
                self.render_flags['all_wireframe'] = True
                self.render_flags['all_solid'] = False
            elif self.render_flags['all_wireframe']:
                self.render_flags['flip_wireframe'] = False
                self.render_flags['all_wireframe'] = False
                self.render_flags['all_solid'] = True
            elif self.render_flags['all_solid']:
                self.render_flags['flip_wireframe'] = False
                self.render_flags['all_wireframe'] = False
                self.render_flags['all_solid'] = False
            else:
                self.render_flags['flip_wireframe'] = True
                self.render_flags['all_wireframe'] = False
                self.render_flags['all_solid'] = False

        # L toggles the lighting mode
        elif symbol == pyglet.window.key.L:
            if self.viewer_flags['use_raymond_lighting']:
                self.viewer_flags['use_raymond_lighting'] = False
                self.viewer_flags['use_direct_lighting'] = True
            elif self.viewer_flags['use_direct_lighting']:
                self.viewer_flags['use_raymond_lighting'] = False
                self.viewer_flags['use_direct_lighting'] = False
            else:
                self.viewer_flags['use_raymond_lighting'] = True
                self.viewer_flags['use_direct_lighting'] = False

        # S toggles shadows
        elif symbol == pyglet.window.key.H:
            self.render_flags['shadows'] = not self.render_flags['shadows']

        # N toggles vertex normals
        elif symbol == pyglet.window.key.N:
            self.render_flags['vertex_normals'] = not self.render_flags['vertex_normals']

        # F toggles face normals
        elif symbol == pyglet.window.key.F:
            self.render_flags['face_normals'] = not self.render_flags['face_normals']

        # Z resets the camera viewpoint
        elif symbol == pyglet.window.key.Z:
            self._reset_view()

        # A causes the frame to rotate
        elif symbol == pyglet.window.key.A:
            self.viewer_flags['rotate'] = not self.viewer_flags['rotate']

        # C toggles backface culling
        elif symbol == pyglet.window.key.C:
            self.render_flags['cull_faces'] = not self.render_flags['cull_faces']

        # S saves the current frame as an image
        elif symbol == pyglet.window.key.S:
            self._save_image()

        # Q quits the viewer
        elif symbol == pyglet.window.key.Q:
            self.on_close()

        # R starts recording frames
        elif symbol == pyglet.window.key.R:
            if self.viewer_flags['record']:
                self._save_gif()
                self.set_caption(self.viewer_flags['window_title'])
            else:
                self.set_caption('{} (RECORDING)'.format(self.viewer_flags['window_title']))
            self.viewer_flags['record'] = not self.viewer_flags['record']

    def _render(self):
        """Render the scene into the framebuffer and flip.
        """
        scene = self.scene
        self._camera_node.matrix = self._trackball.pose.copy()

        # Set lighting
        if self.viewer_flags['use_raymond_lighting']:
            for n in self._raymond_lights:
                if not self.scene.has_node(n):
                    scene.add_node(n, parent_node=self._camera_node)
        else:
            for n in self._raymond_lights:
                if self.scene.has_node(n):
                    self.scene.remove_node(n)

        if self.viewer_flags['use_direct_lighting']:
            if not self.scene.has_node(self._direct_light):
                scene.add_node(self._direct_light, parent_node=self._camera_node)
        else:
            if self.scene.has_node(self._direct_light):
                self.scene.remove_node(self._direct_light)

        flags = RenderFlags.NONE
        if self.render_flags['flip_wireframe']:
            flags |= RenderFlags.FLIP_WIREFRAME
        elif self.render_flags['all_wireframe']:
            flags |= RenderFlags.ALL_WIREFRAME
        elif self.render_flags['all_solid']:
            flags |= RenderFlags.ALL_SOLID

        if self.render_flags['shadows']:
            flags |= (RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT)
        if self.render_flags['vertex_normals']:
            flags |= RenderFlags.VERTEX_NORMALS
        if self.render_flags['face_normals']:
            flags |= RenderFlags.FACE_NORMALS
        if not self.render_flags['cull_faces']:
            flags |= RenderFlags.SKIP_CULL_FACES

        self._renderer.render(self.scene, flags)

    def _reset_view(self):
        """Reset the view to a good initial state.

        The view is initially along the positive x-axis a sufficient distance from the scene.
        """
        centroid = self.scene.centroid
        scale = self.scene.scale

        self._camera_node.matrix = self._default_camera_pose
        self._trackball = Trackball(self._default_camera_pose, self.viewport_size, scale, centroid)

    def _get_save_filename(self, file_exts):
        file_types = {
            'png' : ('png files', '*.png'),
            'jpg' : ('jpeg files', '*.jpg'),
            'gif' : ('gif files', '*.gif'),
            'all' : ('all files', '*'),
        }
        filetypes = [file_types[x] for x in file_exts]
        try:
            root = Tk()
            save_dir = self.viewer_flags['save_directory']
            if save_dir is None:
                save_dir = os.getcwd()
            filename = filedialog.asksaveasfilename(initialdir=save_dir,
                                                    title='Select file save location',
                                                    filetypes=filetypes)
        except:
            return None

        root.destroy()
        if filename == ():
            return None
        return filename

    def _save_image(self):
        filename = self._get_save_filename(['png', 'jpg', 'gif', 'all'])
        if filename is not None:
            self.viewer_flags['save_directory'] = os.path.dirname(filename)
            imageio.imwrite(filename, self._renderer.read_color_buf())

    def _save_gif(self):
        filename = self._get_save_filename(['gif', 'all'])
        if filename is not None:
            self.viewer_flags['save_directory'] = os.path.dirname(filename)
            imageio.mimwrite(filename, self._saved_frames,
                             fps=self.viewer_flags['refresh_rate'],
                             palettesize=128, subrectangles=True)
        self._saved_frames = []

    def _record(self):
        """Save another frame for the GIF.
        """
        self._saved_frames.append(self._renderer.read_color_buf())

    def _rotate(self):
        """Animate the scene by rotating the camera.
        """
        az = self.viewer_flags['rotate_rate'] / self.viewer_flags['refresh_rate']
        self._trackball.rotate(az, self.viewer_flags['rotate_axis'])

    def _render(self):
        """Render the scene into the framebuffer and flip.
        """
        scene = self.scene
        self._camera_node.matrix = self._trackball.pose.copy()

        # Set lighting
        if self.viewer_flags['use_raymond_lighting']:
            for n in self._raymond_lights:
                if not self.scene.has_node(n):
                    scene.add_node(n, parent_node=self._camera_node)
        else:
            for n in self._raymond_lights:
                if self.scene.has_node(n):
                    self.scene.remove_node(n)

        if self.viewer_flags['use_direct_lighting']:
            if not self.scene.has_node(self._direct_light):
                scene.add_node(self._direct_light, parent_node=self._camera_node)
        elif self.scene.has_node(self._direct_light):
            self.scene.remove_node(self._direct_light)

        flags = RenderFlags.NONE
        if self.render_flags['flip_wireframe']:
            flags |= RenderFlags.FLIP_WIREFRAME
        elif self.render_flags['all_wireframe']:
            flags |= RenderFlags.ALL_WIREFRAME
        elif self.render_flags['all_solid']:
            flags |= RenderFlags.ALL_SOLID

        if self.render_flags['shadows']:
            flags |= RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT
        if self.render_flags['vertex_normals']:
            flags |= RenderFlags.VERTEX_NORMALS
        if self.render_flags['face_normals']:
            flags |= RenderFlags.FACE_NORMALS
        if not self.render_flags['cull_faces']:
            flags |= RenderFlags.SKIP_CULL_FACES

        self._renderer.render(self.scene, flags)

    @staticmethod
    def time_event(dt, self):
        if self.viewer_flags['record']:
            self._record()
        if self.viewer_flags['rotate'] and not self.viewer_flags['mouse_pressed']:
            self._rotate()
        self.on_draw()

    def _compute_initial_camera_pose(self):
        centroid = self.scene.centroid
        scale = self.scene.scale

        s2 = 1.0/np.sqrt(2.0)
        cp = np.eye(4)
        cp[:3,:3] = np.array([
            [0.0, -s2,  s2],
            [1.0, 0.0, 0.0],
            [0.0, s2, s2]
        ])
        cp[:3,3] = np.sqrt(2.0)*np.array([scale, 0.0, scale]) + centroid

        return cp

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
        phis = np.pi * np.array([0.0, 2.0/3.0, 4.0/3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            nodes.append(Node(
                light=DirectionalLight(color=np.ones(3), intensity=3.3),
                matrix=matrix
            ))

        return nodes

    def _create_direct_light(self):
        l = DirectionalLight(color=np.ones(3), intensity=10.0)
        n = Node(light=l, matrix=np.eye(4))
        return n
