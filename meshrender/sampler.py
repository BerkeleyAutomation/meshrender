from .constants import GLTF

class Sampler(object):

    def __init__(self,
                 name,
                 magFilter,
                 minFilter,
                 wrapS=GLTF.REPEAT,
                 wrapT=GLTF.REPEAT):
        self.name = name
        self.magFilter = magFilter
        self.minFilter = minFilter
        self.wrapS = wrapS
        self.wrapT = wrapT
