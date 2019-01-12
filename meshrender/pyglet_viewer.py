import os
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
import logging

_USE_EGL_OFFSCREEN = False
if 'MESHRENDER_EGL_OFFSCREEN' in os.environ:
    _USE_EGL_OFFSCREEN = True

import OpenGL

from .constants import OPEN_GL_MAJOR, OPEN_GL_MINOR
from .light import DirectionalLight
from .node import Node
from .camera import PerspectiveCamera
from .trackball import Trackball
from .renderer import Renderer

from autolab_core import transformations, RigidTransform
from perception import CameraIntrinsics, ColorImage

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
    _raymond_lights = None


    def __init__(self, scene, size=(640,480), raymond_lighting=True,
                 animate=False, animate_az=0.05, animate_rate=30.0, animate_axis=None,
                 two_sided_lighting=False, line_width = 1.0,
                 registered_keys={}, starting_camera_pose=None, max_frames=0,
                 save_directory=None, save_filename=None,
                 title='Scene Viewer', target_object=None, **kwargs):
        """Initialize a scene viewer and open the viewer window.

        Parameters
        ----------
        scene : Scene
            A scene to view. The scene's camera is not used.
        size : (int, int)
            The width and height of the target window in pixels.
        raymond_lighting : bool
            If True, the scene's point and directional lights are discarded in favor
            of a set of three directional lights that move with the camera.
        animate : bool
            If True, the camera will rotate by default about the scene.
        animate_az : float
            The number of radians to rotate per timestep.
        animate_rate : float
            The framerate for animation in fps.
        animate_axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        two_sided_lighting : bool
            If True, the shader will treat all normals as facing the camera.
        line_width : float
            Sets the line width for wireframe meshes (default is 1).
        registered_keys : dict
            Map from alphabetic key to a tuple containing
            (1) a callback function taking the viewer itself as its first argument and
            (2) an additional list of arguments for the callback.
        starting_camera_pose : autolab_core.RigidTransform
            An initial pose for the camera, if specified.
        max_frames : int
            If greater than zero, the viewer will display for the given
            number of frames, save those frames, and then close.
        save_directory : str
            A directory to open the TK save file dialog in to begin with.
            If None, uses the current directory.
        save_filename : str
            A default filename to open the save box with. Shouldn't have an extension --
            extension will be .png or .gif depending on save type.
        title : str
            A title for the scene viewer.
        target_object : str
            The name of the object in the scene to center rotations around.
        kwargs : other kwargs
            Other optional keyword arguments.
        """
        if _USE_EGL_OFFSCREEN:
            raise ValueError('Cannot initialize SceneViewer when MESHRENDER_EGL_OFFSCREEN is set.')
        self._gl_initialized = False

        # Save basic information
        self.scene = scene
        self._size = np.array(size)
        self._camera = None         # These two are initialized
        self._trackball = None      # by reset_view()
        self._saved_frames = []

        # Save settings
        self._animate_az = animate_az
        self._animate_rate = animate_rate
        self._animate_axis = animate_axis
        self._line_width = line_width
        self._registered_keys = {
            ord(k.lower()) : registered_keys[k] for k in registered_keys
        }
        self._starting_camera_pose = starting_camera_pose
        self._max_frames = max_frames
        self._save_directory = save_directory
        self._save_filename = save_filename
        self._raymond_lighting = raymond_lighting
        self._raymond_lights = SceneViewer._get_raymond_lights()
        self._title = title
        self._target_object = target_object

        # Set flags
        self._flags = {
            'mouse_pressed' : False,
            'flip_wireframe' : False,
            'two_sided_lighting' : two_sided_lighting,
            'animate' : animate,
            'record' : (self._max_frames > 0),
        }

        # Set up the window
        self._reset_view()
        try:
            conf = gl.Config(sample_buffers=1, samples=4,
                                depth_size=24, double_buffer=True,
                                major_version=OPEN_GL_MAJOR,
                                minor_version=OPEN_GL_MINOR)
            super(SceneViewer, self).__init__(config=conf, resizable=True,
                                                width=self._size[0],
                                                height=self._size[1])
        except Exception as e:
            raise ValueError('Failed to initialize Pyglet window with an OpenGL 3+ context. ' \
                             'If you\'re logged in via SSH, ensure that you\'re running your script ' \
                             'with vglrun (i.e. VirtualGL). Otherwise, the internal error message was: ' \
                             '"{}"'.format(e.message))

        self.set_caption(title)

        self._renderer = Renderer()

        # Update the application flags
        self._update_flags()

        # Start the event loop
        pyglet.app.run()

    @property
    def saved_frames(self):
        """list of perception.ColorImage : Any color images that have been saved
        due to recording or the max_frames argument.
        """
        return [ColorImage(f) for f in self._saved_frames]

    @property
    def save_directory(self):
        """str : A directory to open the TK save file dialog in to begin with.
        """
        return self._save_directory


    def on_close(self):
        """Exit the event loop when the window is closed.
        """
        OpenGL.contextdata.cleanupContext()
        self.close()
        pyglet.app.exit()

    def on_draw(self):
        """Redraw the scene into the viewing window.
        """
        self._render()

    def on_resize(self, width, height):
        """Resize the camera and trackball when the window is resized.
        """
        self._size = (width, height)
        self._trackball.resize(self._size)
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
        self._flags['mouse_pressed'] = True


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Record a mouse drag.
        """
        self._trackball.drag(np.array([x, y]))


    def on_mouse_release(self, x, y, button, modifiers):
        """Record a mouse release.
        """
        self._flags['mouse_pressed'] = False


    def on_mouse_scroll(self, x, y, dx, dy):
        """Record a mouse scroll.
        """
        self._trackball.scroll(dy)

    def on_key_press(self, symbol, modifiers):
        """Record a key press.
        """
        if symbol in self._registered_keys:
            tup = self._registered_keys[symbol]
            callback, args = tup
            callback(self, *args)
        elif symbol == pyglet.window.key.W:
            self._flags['flip_wireframe'] = not self._flags['flip_wireframe']
        elif symbol == pyglet.window.key.Z:
            self._reset_view()
        elif symbol == pyglet.window.key.A:
            self._flags['animate'] = not self._flags['animate']
        elif symbol == pyglet.window.key.L:
            self._flags['two_sided_lighting'] = not self._flags['two_sided_lighting']
        elif symbol == pyglet.window.key.S:
            self._save_image()
        elif symbol == pyglet.window.key.Q:
            self.on_close()
        elif symbol == pyglet.window.key.R:
            if self._flags['record']:
                self._save_gif()
                self.set_caption(self._title)
            else:
                self.set_caption('{} (RECORDING)'.format(self._title))
            self._flags['record'] = not self._flags['record']
        self._update_flags()


    def _update_flags(self):
        """Update OpenGL state based on the current flags.
        """
        clock.set_fps_limit(self._animate_rate)
        clock.unschedule(SceneViewer.time_event)
        if self._flags['animate'] or self._flags['record']:
            clock.schedule_interval(SceneViewer.time_event, 1.0/self._animate_rate, self)


    def _reset_view(self):
        """Reset the view to a good initial state.

        The view is initially along the positive x-axis a sufficient distance from the scene.
        """

        # Compute scene bounds and scale
        bounds = self.scene.bounds
        centroid = self.scene.centroid
        extents = self.scene.extents
        scale = self.scene.scale

        s2 = 1.0/np.sqrt(2.0)
        cp = RigidTransform(
            rotation = np.array([
                [0.0, -s2,  s2],
                [1.0, 0.0, 0.0],
                [0.0, s2, s2]
            ]),
            translation = np.sqrt(2.0)*np.array([scale, 0.0, scale]) + centroid,
            from_frame='camera',
            to_frame='world'
        )

        # Set up reasonable camera intrinsics
        if self.scene.main_camera_node is None:
            camera = PerspectiveCamera(yfov=np.pi / 6.0)
            node = Node(camera=camera, matrix=cp.matrix)
            self.scene.add_node(node)
        else:
            self.scene.main_camera_node.matrix = cp.matrix

        # Create a trackball
        self._trackball = Trackball(
            cp,
            self._size, scale,
            target=centroid,
        )


    def _save_image(self):
        # Get save file location
        try:
            root = Tk()
            fn = ''
            if self._save_filename:
                fn = '{}.png'.format(self._save_filename)
            filename = filedialog.asksaveasfilename(initialfile = fn,
                                                    initialdir = (self._save_directory or os.getcwd()),
                                                    title = 'Select file save location',
                                                    filetypes = (('png files','*.png'),
                                                                ('jpeg files', '*.jpg'),
                                                                ('all files','*.*')))
        except:
            logging.warning('Cannot use Tkinter file dialogs over SSH')
            return

        root.destroy()
        if filename == ():
            return
        else:
            self._save_directory = os.path.dirname(filename)

        # Extract color image from frame buffer
        width, height = self._size
        glReadBuffer(GL_FRONT)
        color_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

        # Re-format them into numpy arrays
        color_im = np.frombuffer(color_buf, dtype=np.uint8).reshape((height, width, 3))
        color_im = np.flip(color_im, axis=0)

        imageio.imwrite(filename, color_im)


    def _save_gif(self):
        # Get save file location
        try:
            root = Tk()
            fn = ''
            if self._save_filename:
                fn = '{}.gif'.format(self._save_filename)
            filename = filedialog.asksaveasfilename(initialfile = fn,
                                                    initialdir = (self._save_directory or os.getcwd()),
                                                    title = 'Select file save location',
                                                    filetypes = (('gif files','*.gif'),
                                                                ('all files','*.*')))
        except:
            logging.warning('Cannot use Tkinter file dialogs over SSH')
            self._saved_frames = []
            return

        root.destroy()
        if filename == ():
            self._saved_frames = []
            return
        else:
            self._save_directory = os.path.dirname(filename)

        imageio.mimwrite(filename, self._saved_frames, fps=30.0, palettesize=128, subrectangles=True)

        self._saved_frames = []


    def _record(self):
        """Save another frame for the GIF.
        """
        # Extract color image from frame buffer
        width, height = self._size
        glReadBuffer(GL_FRONT)
        color_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

        # Re-format them into numpy arrays
        color_im = np.frombuffer(color_buf, dtype=np.uint8).reshape((height, width, 3))
        color_im = np.flip(color_im, axis=0)

        self._saved_frames.append(color_im)

        if self._max_frames:
            if len(self._saved_frames) == self._max_frames:
                self.on_close()


    def _animate(self):
        """Animate the scene by rotating the camera.
        """
        self._trackball.rotate(self._animate_az, self._animate_axis)

    def _render(self):
        """Render the scene into the framebuffer and flip.
        """
        scene = self.scene
        scene.set_pose(self.scene.camera_name, self._trackball.pose)

        if self._raymond_lighting:
            d_lights = []
            for i, dlight in enumerate(SceneViewer._raymond_lights):
                name = '__raymond_light_{}'.format(i)
                if name in scene.directional_lights:
                    scene.set_pose(name, self._trackball.pose)
                else:
                    scene.add(dlight, name=name, pose=self._trackball.pose)

        self._renderer.render(self.scene, 0)

    @staticmethod
    def time_event(dt, self):
        if self._flags['record']:
            self._record()
        if self._flags['animate'] and not self._flags['mouse_pressed']:
            self._animate()

    @staticmethod
    def _get_raymond_lights():
        """Create raymond lights for the scene.
        """
        if SceneViewer._raymond_lights:
            return SceneViewer._raymond_lights

        raymond_lights = []

        # Create raymond lights
        elevs = np.pi * np.array([1/6., 1/6., 1/4.])
        azims = np.pi * np.array([1/6., 5/3., -1/4.])
        l = 0
        for az, el in zip(azims, elevs):
            x = np.cos(el) * np.cos(az)
            y = -np.cos(el) * np.sin(el)
            z = -np.sin(el)

            d = -np.array([x, y, z])
            d = d / np.linalg.norm(d)

            x = np.array([-d[1], d[0], 0.0])
            if np.linalg.norm(x) == 0.0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            l = DirectionalLight(
                color=np.ones(3),
                intensity=0.3,
            )
            matrix = np.eye(4)
            matrix[:3,:3]= np.c_[x,y,z]

            raymond_lights.append(Node(light=l, matrix=matrix))

        SceneViewer._raymond_lights = raymond_lights
        return raymond_lights
