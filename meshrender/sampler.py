from .constants import GLTF

class Sampler(object):

    def __init__(self,
                 name,
                 mag_filter,
                 min_filter,
                 wrap_s=GLTF.REPEAT,
                 wrap_t=GLTF.REPEAT):
        self.name = name
        self.mag_filter = mag_filter
        self.min_filter = min_filter
        self.wrap_s = wrap_s
        self.wrap_t = wrap_t
