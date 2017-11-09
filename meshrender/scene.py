import numpy as np

from perception import BinaryImage, ColorImage, DepthImage, RgbdImage, GdImage, RenderMode

from .camera import VirtualCamera
from .scene_object import SceneObject
from .material import MaterialProperties
from .light import AmbientLight, PointLight, DirectionalLight
from .constants import MAX_N_LIGHTS
from .render import OpenGLRenderer

class Scene(object):
    """A scene containing objects and lights for 3D OpenGL rendering.
    """

    def __init__(self, background_color=np.array([1.0, 1.0, 1.0])):
        """Initialize a Scene object.

        Parameters
        ----------
        background_color : (3,) float
            The background color for the scene.
        """
        self._objects = {}
        self._lights = {}
        self._ambient_light = AmbientLight(np.array([0.,0.,0.]), 0.0)
        self._background_color = background_color

        self._camera = None
        self._renderer = None

    @property
    def background_color(self):
        """(3,) float: The background color for the scene.
        """
        return self._background_color

    @background_color.setter
    def background_color(self, bgcolor):
        self._background_color = bgcolor

    @property
    def objects(self):
        """dict: Dictionary mapping object names to their corresponding SceneObject.
        """
        return self._objects

    @property
    def lights(self):
        """dict: Dictionary mapping light names to their corresponding PointLight or DirectionalLight.

        Note that this doesn't include the ambient light, since only one of those can exist at a time.
        """
        return self._lights

    @property
    def point_lights(self):
        """list of PointLight: The set of point lights active in the scene.
        """
        return [x for x in self.lights.values() if isinstance(x, PointLight)]

    @property
    def directional_lights(self):
        """list of DirectionalLight: The set of directional lights active in the scene.
        """
        return [x for x in self.lights.values() if isinstance(x, DirectionalLight)]

    @property
    def ambient_light(self):
        """AmbientLight: The ambient light active in the scene.
        """
        return self._ambient_light

    @ambient_light.setter
    def ambient_light(self, light):
        if not isinstance(light, AmbientLight):
            raise ValueError('Scene only accepts ambient lights of type AmbientLight')
        self._ambient_light = light

    @property
    def camera(self):
        """VirualCamera: The scene's camera (None if unassigned).
        """
        return self._camera

    @camera.setter
    def camera(self, camera):
        if camera is not None and not isinstance(camera, VirtualCamera):
            raise ValueError('camera must be an object of type VirtualCamera')
        self._camera = camera

    def add_object(self, name, obj):
        """Adds an object to the scene.

        Parameters
        ----------
        name : str
            An identifier for the object.
        obj : SceneObject
            A SceneObject representing the object, including its pose and material properties.
        """
        if not isinstance(obj, SceneObject):
            raise ValueError('obj must be an object of type SceneObject')
        self._objects[name] = obj
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None

    def remove_object(self, name):
        """Removes an object from the scene.

        Parameters
        ----------
        name : str
            An identifier for the object to be removed.

        Raises
        ------
        ValueError
            If the given name was not assigned to an object in the scene.
        """
        if name in self._objects:
            del self._objects[name]
        else:
            raise ValueError('Object {} not in scene!'.format(name))
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None

    def add_light(self, name, light):
        """Adds a named light to the scene.

        Parameters
        ----------
        name : str
            An identifier for the light.
        light : PointLight or DirectionalLight
            The light source to add.
        """
        if isinstance(light, AmbientLight):
            raise ValueError('Set ambient light with set_ambient_light(), not with add_light()')
        if len(self._lights) == MAX_N_LIGHTS:
            raise ValueError('The maximum number of lights in a scene is capped at {}'.format(MAX_N_LIGHTS))
        if not isinstance(light, PointLight) and not isinstance(light, DirectionalLight):
            raise ValueError('Scene only supports PointLight and DirectionalLight types')
        self._lights[name] = light

    def remove_light(self, name):
        """Removes a light from the scene.

        Parameters
        ----------
        name : str
            An identifier for the light to be removed.

        Raises
        ------
        ValueError
            If the given name was not assigned to a light in the scene.
        """
        if name in self._lights:
            del self._lights[name]
        else:
            raise ValueError('Light {} not in scene!'.format(name))

    def close_renderer(self):
        """Close the renderer.
        """
        if self._renderer is not None:
            self._renderer.close()

    def render(self, render_color=True, front_and_back=False):
        """Render raw images of the scene.

        Parameters
        ----------
        render_color : bool
            If True, both a color and a depth image are returned.
            If False, only a depth image is returned.

        front_and_back : bool
            If True, all surface normals are treated as if they're facing the camera.

        Returns
        -------
        tuple of (h, w, 3) uint8, (h, w) float32
            A raw RGB color image with pixel values in [0, 255] and a depth image
            with true depths expressed as floats. If render_color was False,
            only the depth image is returned.

        Raises
        ------
        ValueError
            If the scene has no set camera.

        Note
        -----
        This function can be called repeatedly, regardless of changes to the scene
        (i.e. moving SceneObjects, adding and removing lights, moving the camera).
        However, adding or removing objects causes a new OpenGL context to be created,
        so put all the objects in the scene before calling it.

        Note
        ----
        Values listed as 0.0 in the depth image are actually at infinity
        (i.e. no object present at that pixel).
        """
        if self._camera is None:
            raise ValueError('scene.camera must be set before calling render()')
        if self._renderer is None:
            self._renderer = OpenGLRenderer(self)
        return self._renderer.render(render_color, front_and_back=front_and_back)

    def wrapped_render(self, render_modes, front_and_back=False):
        """Render images of the scene and wrap them with Image wrapper classes
        from the Berkeley AUTOLab's perception module.

        Parameters
        ----------
        render_modes : list of perception.RenderMode 
            A list of the desired image types to return, from the perception
            module's RenderMode enum.

        front_and_back : bool
            If True, all surface normals are treated as if they're facing the camera.

        Returns
        -------
        list of perception.Image
            A list containing a corresponding Image sub-class for each type listed
            in render_modes.
        """

        # Render raw images
        render_color = False
        for mode in render_modes:
            if mode != RenderMode.DEPTH and mode != RenderMode.SCALED_DEPTH:
                render_color = True
                break

        color_im, depth_im = None, None
        if render_color:
            color_im, depth_im = self.render(render_color, front_and_back=front_and_back)
        else:
            depth_im = self.render(render_color)

        # For each specified render mode, add an Image object of the appropriate type
        images = []
        for render_mode in render_modes:
            # Then, convert them to an image wrapper class
            if render_mode == RenderMode.SEGMASK:
                images.append(BinaryImage((depth_im > 0.0).astype(np.uint8), frame=self.camera.intrinsics.frame, threshold=0))

            elif render_mode == RenderMode.COLOR:
                images.append(ColorImage(color_im, frame=self.camera.intrinsics.frame))

            elif render_mode == RenderMode.GRAYSCALE:
                images.append(ColorImage(color_im, frame=self.camera.intrinsics.frame).to_grayscale())

            elif render_mode == RenderMode.DEPTH:
                images.append(DepthImage(depth_im, frame=self.camera.intrinsics.frame))

            elif render_mode == RenderMode.SCALED_DEPTH:
                images.append(DepthImage(depth_im, frame=self.camera.intrinsics.frame).to_color())

            elif render_mode == RenderMode.RGBD:
                c = ColorImage(color_im, frame=self.camera.intrinsics.frame)
                d = DepthImage(depth_im, frame=self.camera.intrinsics.frame)
                images.append(RgbdImage.from_color_and_depth(c, d))

            elif render_mode == RenderMode.GD:
                g = ColorImage(color_im, frame=self.camera.intrinsics.frame).to_grayscale()
                d = DepthImage(depth_im, frame=self.camera.intrinsics.frame)
                images.append(GdImage.from_grayscale_and_depth(g, d))
            else:
                raise ValueError('Render mode {} not supported'.format(render_mode))

        return images

    def __del__(self):
        if self._renderer is not None:
            self._renderer.close()
