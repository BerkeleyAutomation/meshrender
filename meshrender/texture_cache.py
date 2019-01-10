import hashlib
import numpy as np
import uuid

from .texture import Texture

class TextureCache(object):
    """A cache for textures.
    """

    def __init__(self):
        self._name_cache = {}
        self._hash_cache = {}

    def get_texture(self, name=None, sampler=None,
                    source=None, source_channels=None,
                    width=None, height=None):
        """Get a texture from a numpy array.

        Parameters
        ----------
        data : np.ndarray
            Array containing texture data.
        """
        if name is not None:
            if name in self._name_cache:
                return self._name_cache[name]
        else:
            name = uuid.uuid4().hex

        # Hash data to find shader
        if data is not None:
            data = np.ascontiguousarray(source)
            hasher = hashlib.md5(data)
            md5 = hasher.hexdigest()
            if md5 in self._hash_cache:
                tex = self._hash_cache[md5]
                if name not in self._name_cache:
                    self._name_cache[name] = tex
                return tex

        # If none found, create shader
        tex = Texture(
            data=data,
            sampler=sampler,
            source=source,
            source_channels=source_channels,
            width=width,
            height=height
        )

        tex._add_to_context()

        self._name_cache[name] = tex
        self._hash_cache[md5] = tex

        return tex

    @property
    def texture_names(self):
        return set(self._name_cache.keys())

    @property
    def cache_size(self):
        return len(self._hash_cache)

    def clear(self):
        self.delete()

    def delete(self):
        """Delete all cached shader programs.
        """
        for key in self._hash_cache:
            self._hash_cache[key].delete()
        self._name_cache = {}
        self._hash_cache = {}

    def delete_texture(self, name):
        if name in self._name_cache:
            tex = self._name_cache.pop(key)
            for k in self._hash_cache:
                if self._hash_cache[k] == tex:
                    self._hash_cache.pop(k)
            tex.delete()

