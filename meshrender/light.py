"""A set of all allowed lights in a Scene.
"""
import numpy as np

class Light(object):
    """Base class for all light objects.
    """

    def __init__(self, color, strength):
        """Initialize a light with the given color and strength.

        Parameters
        ----------
        color : (3,) float
            The RGB color of the light in (0,1).
        strength : float
            The strength of the light.
        """
        self._color = color
        self._strength = strength

    @property
    def color(self):
        """(3,) float: The RGB color of the light in (0,1).
        """
        return self._color

    @property
    def strength(self):
        """float: The strength of the light.
        """
        return self._strength

class AmbientLight(Light):
    """An ambient light, which flatly shades all objects in the world.
    """

    def __init__(self, color, strength):
        """Initialize an ambient light with the given color and strength.

        Parameters
        ----------
        color : (3,) float
            The RGB color of the light in (0,1).
        strength : float
            The strength of the light.
        """
        super(AmbientLight, self).__init__(color, strength)


class DirectionalLight(Light):
    """A far-away light with a given direction.
    """

    def __init__(self, direction, color, strength):
        """Initialize a directional light with the given direction, color, and strength.

        Parameters
        ----------
        direction : (3,) float
            A unit vector indicating the direction of the light.
        color : (3,) float
            The RGB color of the light in (0,1).
        strength : float
            The strength of the light.
        """
        self._direction = direction
        super(DirectionalLight, self).__init__(color, strength)

    @property
    def direction(self):
        """(3,) float: A unit vector indicating the direction of the light.
        """
        return self._direction

    @direction.setter
    def direction(self, d):
        self._direction = d


class PointLight(Light):
    """A nearby point light source that shines in all directions.
    """

    def __init__(self, location, color, strength):
        """Initialize a point light with the given location, color, and strength.

        Parameters
        ----------
        location : (3,) float
            The 3D location of the point light.
        color : (3,) float
            The RGB color of the light in (0,1).
        strength : float
            The strength of the light.
        """
        self._location = location
        super(PointLight, self).__init__(color, strength)

    @property
    def location(self):
        """(3,) float: The 3D location of the point light.
        """
        return self._location

    @location.setter
    def location(self, l):
        self._location = l
