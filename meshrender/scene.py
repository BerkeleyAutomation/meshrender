import numpy as np

from .light import AmbientLight, PointLight, DirectionalLight

class Scene(object):

    def __init__(self):
        self._objects = {}
        self._lights = {}
        self._ambient_light = AmbientLight(np.array([0.,0.,0.]), 0.0)
        self._camera = None

    def add_object(self, name, geometry, transform, material):
        self._objects[name] = {
            'geometry' : geometry,
            'transform' : transform,
            'material' : material
        }

    def remove_object(self, name):
        if name in self._objects:
            del self._objects[name]
        else:
            raise ValueError('Object {} not in scene!'.format(name))

    def set_transform(self, name, transform):
        if name in self._objects:
            self._objects[name]['transform'] = transform
        else:
            raise ValueError('Object {} not in scene!'.format(name))

    def get_transform(self, name):
        if name in self._objects:
            return self._objects[name]['transform']
        else:
            raise ValueError('Object {} not in scene!'.format(name))

    @property
    def object_keys(self):
        return self._objects.keys()

    @property
    def objects(self):
        return self._objects.values()

    @property
    def light_keys(self):
        return self._lights.keys()

    @property
    def lights(self):
        return self._lights.values()

    @property
    def point_lights(self):
        return [x for x in self.lights if isinstance(x, PointLight)]

    @property
    def directional_lights(self):
        return [x for x in self.lights if isinstance(x, DirectionalLight)]

    @property
    def ambient_light(self):
        return self._ambient_light

    @property
    def camera(self):
        return self._camera

    def add_camera(self, camera):
        self._camera = camera

    def set_ambient_light(self, light):
        if not isinstance(light, AmbientLight):
            raise ValueError('Scene only accepts ambient lights from class AmbientLight!')
        self._ambient_light = light

    def add_light(self, name, light):
        if isinstance(light, AmbientLight):
            raise ValueError('Set ambient light with set_ambient_light(), not with add_light()!')
        self._lights[name] = light

    def remove_light(self, name):
        if name in self._lights:
            del self._lights[name]
        else:
            raise ValueError('Light {} not in scene!'.format(name))

    def render_rgb(self):
        pass

    def render_depth(self):
        pass
