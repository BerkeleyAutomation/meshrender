'''A set of possible lights in a scene.
'''

import numpy as np

class Light(object):

    def __init__(self, color, strength):
        '''Initialize a light with the given color and strength

        Parameters
        ----------
        color:     (3,) float, the RGB color of the light in (0,1)
        strength:       float, the strength of the light
        '''
        self._color = color
        self._strength = strength

    @property
    def color(self):
        return self._color

    @property
    def strength(self):
        return self._strength

class AmbientLight(Light):

    def __init__(self, color, strength):
        '''Initialize an ambient light with the given color and strength.

        Parameters
        ----------
        color:     (3,) float, the RGB color of the light in (0,1)
        strength:       float, the strength of the light
        '''
        super(AmbientLight, self).__init__(color, strength)


class DirectionalLight(Light):

    def __init__(self, direction, color, strength):
        '''Initialize a directional light with the given direction, color, and strength.

        Parameters
        ----------
        direction: (3,) float, the direction of the light
        color:     (3,) float, the RGB color of the light in (0,1)
        strength:       float, the strength of the light
        '''
        self._direction = direction
        super(DirectionalLight, self).__init__(color, strength)

    @property
    def direction(self):
        return self._direction


class PointLight(Light):

    def __init__(self, location, color, strength):
        '''Initialize a point light with the given location, color, and strength.

        Parameters
        ----------
        location: (3,) float, the location of the light
        color:    (3,) float, the RGB color of the light in (0,1)
        strength:      float, the strength of the light
        '''
        self._location = location
        super(PointLight, self).__init__(color, strength)

    @property
    def location(self):
        return self._location
