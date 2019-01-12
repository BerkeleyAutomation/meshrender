from .constants import GLTF

class Sampler(object):

    def __init__(self,
                 name=None,
                 magFilter=None,
                 minFilter=None,
                 wrapS=GLTF.REPEAT,
                 wrapT=GLTF.REPEAT):
        self.name = name
        self.magFilter = magFilter
        self.minFilter = minFilter
        self.wrapS = wrapS
        self.wrapT = wrapT
