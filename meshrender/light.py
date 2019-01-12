"""Punctual light sources as defined by the glTF 2.0 KHR extension.
https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md
"""
import abc
import numpy as np
import six

from .utils import format_color_vector
from .texture import Texture

@six.add_metaclass(abc.ABCMeta)
class Light(object):
    """Base class for all light objects.

    Attributes
    ----------
    name : str, optional
        Name for the light.
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of the light. Point and Spot lights define
        this in candela (lm/sr), while directional lights use lux (lm/m^2).
    range : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, range is infinite.
    """
    def __init__(self,
                 name=None,
                 color=None,
                 intensity=None,
                 range=None,
                 castsShadows=False):
        self.name = name
        self.color = color
        self.intensity = intensity
        self.range = range
        self.castsShadows = castsShadows
        self.depth_texture = None
        self.depth_camera = None

        if self.castsShadows:
            self.depth_texture = self._generate_depth_texture()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = format_color_vector(value, 3)

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = float(value)

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, value):
        if value is None or value < 0.0:
            self._range = value
        else:
            self._range = float(value)

    @abc.abstractmethod
    def _generate_depth_texture(self):
        pass

class DirectionalLight(Light):
    """A directional light source, which casts light along a given
    direction with parallel rays.
    Light is emitted in the positive z-direction in the local frame.

    Attributes
    ----------
    name : str, optional
        Name for the light.
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of the light. Point and Spot lights define
        this in candela (lm/sr), while directional lights use lux (lm/m^2).
    range : float
        Unused for directional lights.
    """

    def __init__(self,
                 name=None,
                 color=None,
                 intensity=None,
                 range=None):
        super(DirectionalLight, self).__init__(
            name=name,
            color=color,
            intensity=intensity,
            range=range
        )

    def _generate_depth_texture(self):
        return Texture(width=1024, height=1024, source_channels='D')

class PointLight(Light):
    """A point light source, which casts light in all directions and attenuates with
    distance.

    The attenuation is computed as

    F_att = 1.0 / (K_c + K_l * d + K_q * d^2),

    where K_c is the constant term, K_l is the linear term,
    and K_q is the quadratic term.

    Attributes
    ----------
    name : str, optional
        Name for the light.
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of the light. Point and Spot lights define
        this in candela (lm/sr), while directional lights use lux (lm/m^2).
    range : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, range is infinite.
    """

    def __init__(self,
                 name=None,
                 color=None,
                 intensity=None,
                 range=None):
        super(PointLight, self).__init__(
            name=name,
            color=color,
            intensity=intensity,
            range=range
        )

    def _generate_depth_texture(self):
        raise NotImplementedError('Shadows not yet implemented for point lights')

class SpotLight(Light):
    """A spot light source, which casts light in a particular direction in a cone.

    Attributes
    ----------
    name : str, optional
        Name for the light.
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of the light. Point and Spot lights define
        this in candela (lm/sr), while directional lights use lux (lm/m^2).
    range : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, range is infinite.
    innerConeAngle : float
        Inner cutoff angle, in radians.
    outerConeAngle : float
        Outer cutoff angle, in radians.
    """

    def __init__(self,
                 name=None,
                 color=None,
                 intensity=None,
                 range=None,
                 innerConeAngle=0.0,
                 outerConeAngle=np.pi/4.0):
        super(PointLight, self).__init__(
            name=name,
            color=color,
            intensity=intensity,
            range=range
        )
        self.outerConeAngle = outerConeAngle
        self.innerConeAngle = innerConeAngle

    @property
    def innerConeAngle(self):
        return self._innerConeAngle

    @innerConeAngle.setter
    def innerConeAngle(self, value):
        if value < 0.0 or value > self.outerConeAngle:
            raise ValueError('Invalid value for inner cone angle')
        self._innerConeAngle = float(value)

    @property
    def outerConeAngle(self):
        return self._outerConeAngle

    @outerConeAngle.setter
    def outerConeAngle(self, value):
        if value < 0.0 or value > np.pi / 2.0 + 1e-9:
            raise ValueError('Invalid value for outer cone angle')
        self._outerConeAngle = float(value)

    def _generate_depth_texture(self):
        return Texture(width=1024, height=1024, source_channels='D')

