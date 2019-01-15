import numpy as np

from OpenGL.GL import *

from .utils import format_texture_source
from .sampler import Sampler

class Texture(object):

    def __init__(self,
                 name=None,
                 sampler=None,
                 source=None,
                 source_channels=None,
                 width=None,
                 height=None,
                 config=GL_TEXTURE_2D):
        self.source_channels = source_channels
        self.name = name
        self.sampler = sampler
        self.source = source
        self.width = width
        self.height = height
        self.config = config

        self._texid = None

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        if value is None:
            value = Sampler()
        self._sampler = value

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        if value is None:
            self._source = None
        else:
            self._source = format_texture_source(value, self.source_channels)

    ##################
    # OpenGL code
    ##################
    def _add_to_context(self):
        if self._texid is not None:
            raise ValueError('Texture already loaded into OpenGL context')

        fmt = GL_DEPTH_COMPONENT
        if self.source_channels == 'R':
            fmt = GL_RED
        elif self.source_channels == 'RG' or self.source_channels == 'GB':
            fmt = GL_RG
        elif self.source_channels == 'RGB':
            fmt = GL_RGB
        elif self.source_channels == 'RGBA':
            fmt = GL_RGBA

        # Generate the OpenGL texture
        self._texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texid)

        # Flip data for OpenGL buffer
        data = None
        width = self.width
        height = self.height
        if self.source is not None:
            data = np.ascontiguousarray(np.flip(self.source, axis=0).flatten())
            width = self.source.shape[1]
            height = self.source.shape[0]

        # Bind texture and generate mipmaps
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, width, height, 0, fmt, GL_FLOAT, data)
        if self.source is not None:
            glGenerateMipmap(GL_TEXTURE_2D)
            if self.sampler.magFilter is not None:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, self.sampler.magFilter)
            if self.sampler.minFilter is not None:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, self.sampler.minFilter)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, self.sampler.wrapS)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, self.sampler.wrapT)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, np.ones(4).astype(np.float32))

        # Unbind texture
        glBindTexture(GL_TEXTURE_2D, 0)

    def _remove_from_context(self):
        if self._texid is not None:
            # TODO OPENGL BUG?
            #glDeleteTextures(1, [self._texid])
            glDeleteTextures([self._texid])
            self._texid = None

    def _in_context(self):
        return self._texid is not None

    def _bind(self):
        # TODO HANDLE INDEXING INTO OTHER UV's
        glBindTexture(GL_TEXTURE_2D, self._texid)

    def _unbind(self):
        glBindTexture(GL_TEXTURE_2D, 0)

    def _bind_as_depth_attachment(self):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self._texid, 0)

    def _bind_as_color_attachment(self):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._texid, 0)

    def delete(self):
        self._unbind()
        self._remove_from_context()
