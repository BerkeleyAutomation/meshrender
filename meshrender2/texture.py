import hashlib
import uuid

class TextureCache(object):
    """A cache for textures.
    """

    def __init__(self):
        self._name_cache = {}
        self._hash_cache = {}

    def get_texture(self, name=None, data=None,
                    width=None, height=None, n_components=None):
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
            data = np.ascontiguousarray(data)
            hasher = hashlib.md5(contiguous)
            md5 = hasher.hexdigest()
            if md5 in self._hash_cache:
                tex = self._hash_cache[md5]
                if name not in self._name_cache:
                    self._name_cache[name] = tex
                return tex

        # If none found, create shader
        tex = Texture(data=data, width=width, height=height,
                      n_components=n_components)

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
        for key in self._hash_cache::
            self._hash_cache[key].delete()
        self._name_cache = {}
        self._hash_cache = {}

    def delete_texture(self, name):
        if name in self._name_cache:
            tex = self._name_cache.pop(key)
            for k in self._hash_cache:
                if self._hash_cache[k] == tex:
                    self._hash_cache.pop(k)

class Texture(object):

    def __init__(self, data=None, width=None, height=None, n_components=None):

        if data is not None:
            # Process Data Format
            width = data.shape[1]
            height = data.shape[0]

            if data.ndim == 2:
                if data.dtype == np.float:
                    fmt = GL_DEPTH_COMPONENT
                else:
                    fmt = GL_RED
            elif data.ndim == 3:
                if data.shape[2] == 1:
                    if data.dtype == np.float:
                        fmt = GL_DEPTH_COMPONENT
                    else:
                        fmt = GL_RED
                elif data.shape[2] == 2:
                    fmt = GL_RG
                elif data.shape[2] == 3:
                    fmt = GL_RGB
                elif data.shape[2] == 4:
                    fmt = GL_RGBA
                else:
                    raise ValueError('Unsupported data shape for texture')
            else:
                raise ValueError('Unsupported data shape for texture')

            if data.dtype == np.uint8:
                dfmt = GL_UNSIGNED_BYTE
            elif data.dtype == np.float32:
                dfmt = GL_FLOAT

            data = data.flatten()
        else:
            width = width
            height = height

            fmt = GL_DEPTH_COMPONENT
            if n_components == 1:
                fmt = GL_RED
            elif n_components == 2:
                fmt = GL_RG
            elif n_components == 3:
                fmt = GL_RGB
            elif n_components == 4:
                fmt = GL_RGBA
            else:
                raise ValueError('Unsupported number of components for texture.')
            dfmt = GL_FLOAT


        # Generate the OpenGL texture
        self._texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texid)
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, width, height, 0, fmt, dfmt, data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self._texid)

    def unbind(self):
        glBindTexture(GL_TEXTURE_2D, 0)

    def bind_as_depth_attachment(self):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self._texid, 0)

    def bind_as_color_attachment(self):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._texid, 0)

    def delete(self):
        if self._texid is not None:
            glDeleteTextures(1, self._texid)
            self._texid = None

    def __del__(self):
        self.delete()
