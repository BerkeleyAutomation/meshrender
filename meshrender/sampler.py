from .constants import GLTF

class Sampler(object):

    def __init__(self,
                 name=None,
                 magFilter=None,
                 minFilter=None,
                 #wrapS=GLTF.REPEAT,
                 #wrapT=GLTF.REPEAT):
                 wrapS=GLTF.MIRRORED_REPEAT,
                 wrapT=GLTF.MIRRORED_REPEAT):
        self.name = name
        #self.magFilter = magFilter
        #self.minFilter = minFilter
        self.magFilter = GLTF.LINEAR
        self.minFilter = GLTF.LINEAR_MIPMAP_LINEAR
        self.wrapS = wrapS
        self.wrapT = wrapT
