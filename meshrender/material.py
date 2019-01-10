import abc
import numpy as np
import PIL
import six

@six.add_metaclass(abc.ABCMeta)
class Material(object):
    """Base for standard glTF 2.0 materials.
    """

    def __init__(self,
                 name=None,
                 normal_texture=None,
                 occlusion_texture=None,
                 emissive_texture=None,
                 emissive_factor=None,
                 alpha_mode=None,
                 alpha_cutoff=None,
                 double_sided=False):

        # Set defaults
        if alpha_mode is None:
            alpha_mode = 'OPAQUE'

        if alpha_cutoff is None:
            alpha_cutoff = 0.5

        self.name = name
        self.normal_texture = normal_texture
        self.occlusion_texture = occlusion_texture
        self.emissive_texture = emissive_texture
        self.emissive_factor = emissive_factor
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff
        self.double_sided = double_sided

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def normal_texture(self):
        return self._normal_texture

    @normal_texture.setter
    def normal_texture(self, value):
        self._normal_texture = self._format_texture(value, 'RGB')

    @property
    def occlusion_texture(self):
        return self._occlusion_texture

    @occlusion_texture.setter
    def occlusion_texture(self, value):
        self._occlusion_texture = self._format_texture(value, 'R')

    @property
    def emissive_texture(self):
        return self._emissive_texture

    @emissive_texture.setter
    def emissive_texture(self, value):
        self._emissive_texture = self._format_texture(value, 'RGB')

    @property
    def alpha_mode(self):
        return self._alpha_mode

    @alpha_mode.setter
    def alpha_mode(self, value):
        if value not in set(['OPAQUE', 'MASK', 'BLEND']):
            raise ValueError('Invalid alpha mode {}'.format(value))
        self._alpha_mode = alpha_mode

    @property
    def alpha_cutoff(self):
        return self._alpha_cutoff

    @alpha_cutoff.setter
    def alpha_cutoff(self, value):
        if value < 0 or value > 1:
            raise ValueError('Alpha cutoff must be in range [0,1]')
        self._alpha_cutoff = float(value)

    @property
    def double_sided(self):
        return self._double_sided

    @double_sided.setter
    def double_sided(self, value):
        if not isinstance(value, bool):
            raise TypeError('Double sided must be a boolean value')
        self._double_sided = double_sided

    def _format_texture(self, texture, target_channels='RGB'):
        """Format a texture as a float32 np array.
        """

        # Pass through None
        if texture is None:
            return None

        # Convert PIL images into numpy arrays
        if isinstance(texture, PIL.Image.Image):
            texture = np.array(texture)

        # Format numpy arrays
        if isinstance(texture, np.ndarray):
            if np.issubdtype(texture.dtype, np.integer):
                texture = (texture / 255.0).astype(np.float32)
            elif np.issubdtype(texture.dtype, np.float):
                texture = texture.astype(np.float32)
            else:
                raise TypeError('Invalid type {} for texture'.format(type(texture)))

            # Format array by picking out correct texture channels or padding
            if texture.ndim == 2:
                texture = texture[:,:,np.newaxis]

            if target_channels == 'R':
                texture = texture[:,:,0]
                texture = texture.squeeze()
            elif target_channels == 'RG':
                if texture.shape[2] == 1:
                    texture = np.repeat(texture, 2, axis=2)
                else:
                    texture = texture[:,:,(0,1)]
            elif target_channels == 'GB':
                if texture.shape[2] == 1:
                    texture = np.repeat(texture, 2, axis=2)
                elif texture.shape[2] > 2:
                    texture = texture[:,:,(1,2)]
            elif target_channels == 'RGB':
                if texture.shape[2] == 1:
                    texture = np.repeat(texture, 3, axis=2)
                elif texture.shape[2] == 2:
                    raise ValueError('Cannot reformat texture with 2 channels into RGB')
                else:
                    texture = texture[:,:,(1,2,3)]
            elif target_channels == 'RGBA':
                if texture.shape[2] == 1:
                    texture = np.repeat(texture, 4, axis=2)
                    texture[:,:,3] = 1.0
                elif texture.shape[2] == 2:
                    raise ValueError('Cannot reformat texture with 2 channels into RGBA')
                elif texture.shape[2] == 3:
                    texture = np.concatenate((texture,
                                                np.ones(texture.shape[0],
                                                        texture.shape[1],
                                                        target_n_channels - texture.shape[2])), axis=2)
            else:
                raise ValueError('Invalid texture channel specification: {}'.format(target_channels))
        else:
            raise TypeError('Invalid type {} for texture'.format(type(texture)))

        return texture.astype(np.float32)



class MetallicRoughnessMaterial(Material):

    def __init__(self,
                 name=None,
                 normal_texture=None,
                 occlusion_texture=None,
                 emissive_texture=None,
                 emissive_factor=None,
                 alpha_mode=None,
                 alpha_cutoff=None,
                 double_sided=False,
                 base_color_factor=None,
                 base_color_texture=None,
                 metallic_factor=1.0,
                 roughness_factor=1.0,
                 metallic_roughness_texture=None):
        super(MetallicRoughnessMaterial, self).__init__(
            name=name,
            normal_texture=normal_texture,
            occlusion_texture=occlusion_texture,
            emissive_texture=emissive_texture,
            emissive_factor=emissive_factor,
            alpha_mode=alpha_mode,
            alpha_cutoff=alpha_cutoff,
            double_sided=double_sided
        )

        # Set defaults
        if base_color_factor is None:
            base_color_factor = np.ones(4).astype(np.float32)

        self.base_color_factor = base_color_factor
        self.base_color_texture = base_color_texture
        self.metallic_factor = metallic_factor
        self.roughness_factor = roughness_factor
        self.metallic_roughness_texture = metallic_roughness_texture

    @property
    def base_color_factor(self):
        return self._base_color_factor

    @base_color_factor.setter
    def base_color_factor(self, value):
        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.shape[0] not in set([3,4]):
            raise TypeError('Base color factor must be a (3,) or (4,) ndarray')

        if np.issubdtype(value.dtype, np.integer):
            value = (value / 255.0).astype(np.float32)
        elif np.issubdtype(value.dtype, np.float):
            value = value.astype(np.float32)
        else:
            raise ValueError('Base color factor must be a numerical ndarray')

        self._base_color_factor = base_color_factor

    @property
    def base_color_texture(self):
        return self._base_color_texture

    @base_color_texture.setter
    def base_color_texture(self, value):
        self._base_color_texture = self._format_texture(value, 'RGBA')

    @property
    def metallic_factor(self):
        return self._metallic_factor

    @metallic_factor.setter
    def metallic_factor(self, value):
        if value < 0 or value > 1:
            raise ValueError('Metallic factor must be in range [0,1]')
        self._metallic_factor = float(value)

    @property
    def roughness_factor(self):
        return self._roughness_factor

    @roughness_factor.setter
    def roughness_factor(self, value):
        if value < 0 or value > 1:
            raise ValueError('roughness factor must be in range [0,1]')
        self._roughness_factor = float(value)

    @property
    def metallic_roughness_texture(self):
        return self._metallic_roughness_texture

    @metallic_roughness_texture.setter
    def metallic_roughness_texture(self, value):
        self._metallic_roughness_texture = self._format_texture(value, 'BG')
