"""Lighting options
"""
import numpy as np

class Light(object):
    """Base class for all light objects.

    Attributes
    ----------
    ambient : (3,) float
        The ambient color of the light.
    diffuse : (3,) float
        The diffuse color of the light.
    specular : (3,) float
        The specular color of the light.
    """

    def __init__(self, ambient, diffuse, specular):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

class DirectionalLight(Light):
    """A directional light source, which casts light along a given\
    direction with parallel rays.

    Attributes
    ----------
    ambient : (3,) float
        The ambient color of the light.
    diffuse : (3,) float
        The diffuse color of the light.
    specular : (3,) float
        The specular color of the light.
    direction : (3,) float
        The direction along which the light travels in world frame.
    """

    def __init__(self, ambient, diffuse, specular, direction):
        super(DirectionalLight, self).__init__(ambient, diffuse, specular)
        self.direction = direction

class PointLight(Light):
    """A point light source, which casts light in all directions and attenuates with
    distance.

    The attenuation is computed as

    F_att = 1.0 / (K_c + K_l * d + K_q * d^2),

    where K_c is the constant term, K_l is the linear term,
    and K_q is the quadratic term.

    Attributes
    ----------
    ambient : (3,) float
        The ambient color of the light.
    diffuse : (3,) float
        The diffuse color of the light.
    specular : (3,) float
        The specular color of the light.
    position : (3,) float
        The position of the light in world frame.
    constant : float
        Constant attenuation term.
    linear : float
        Linear attenuation term.
    quadratic : float
        Quadratic attenuation term.
    """

    def __init__(self, ambient, diffuse, specular, position, constant, linear, quadratic):
        super(PointLight, self).__init__(ambient, diffuse, specular)
        self.position = position
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic

class SpotLight(Light):
    """A spot light source, which casts light in a particular direction in a cone.

    The attenuation is computed as

    F_att = 1.0 / (K_c + K_l * d + K_q * d^2),

    where K_c is the constant term, K_l is the linear term,
    and K_q is the quadratic term.

    Light intensity is computed with the following formula:

    theta = dot(light_dir, -direction) # light_dir is vector from fragment to light
    epsilon = inner_angle - outer_angle # (in radians)
    intensity = clamp((theta - outer_angle) / epsilon, 0.0, 1.0)

    Attributes
    ----------
    ambient : (3,) float
        The ambient color of the light.
    diffuse : (3,) float
        The diffuse color of the light.
    specular : (3,) float
        The specular color of the light.
    position : (3,) float
        The position of the light in world frame.
    direction : (3,) float
        The direction of the light in world frame.
    constant : float
        Constant attenuation term.
    linear : float
        Linear attenuation term.
    quadratic : float
        Quadratic attenuation term.
    inner_angle : float
        Inner cutoff angle, in degrees.
    outer_angle : float
        Outer cutoff angle, in degrees.
    """

    def __init__(self, ambient, diffuse, specular, position, direction,
                 constant, linear, quadratic, inner_angle, outer_angle):
        super(SpotLight, self).__init__(ambient, diffuse, specular)
        self.position = position
        self.direction = direction
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic
        self.inner_angle = inner_angle
        self.outer_angle = outer_angle
