"""Material properties, conforming to the glTF 2.0 standard.
https://git.io/fhkPZ
"""
import abc
import numpy as np
import PIL
import six

from .constants import TexFlags
from .utils import format_color_vector, format_texture_source

@six.add_metaclass(abc.ABCMeta)
class Material(object):
    """Base for standard glTF 2.0 materials.

    Attributes
    ----------
    name : str, optional
        The user-defined name of this object.
    normal_texture : (n,n,3) float, optional
        A tangent-space normal map. The texture contains RGB components in linear space.
        Red maps X to [-1,1], green maps Y to [-1, 1] and blue maps Z to [-1, 1].
    occlusion_texture : (n,n,1) float, optional
        The occlusion texture map. If channel size is >1, occlusion values are sampled
        from the R channel.
    emissive_texture : (n,n,3) float, optional
        The emissive lighting map. Colors should be specified in sRGB space.
    emissive_factor : (3,) float
        The RGB components of the emissive color of the material in linear space.
    alpha_mode : str, optional
        One of 'OPAQUE', 'MASK', or 'BLEND'.
    alpha_cutoff : float, optional
        Specifies cutoff threshold when in MASK mode.
    double_sided : bool, optional
        If True, the material is double sided. If False, back-face culling is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal per vertex.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    """

    def __init__(self,
                 name=None,
                 normal_texture=None,
                 occlusion_texture=None,
                 emissive_texture=None,
                 emissive_factor=None,
                 alpha_mode=None,
                 alpha_cutoff=None,
                 double_sided=False,
                 smooth=True,
                 wireframe=False):

        # Set defaults
        if alpha_mode is None:
            alpha_mode = 'OPAQUE'

        if alpha_cutoff is None:
            alpha_cutoff = 0.5

        if emissive_factor is None:
            emissive_factor = np.zeros(3).astype(np.float32)

        self.name = name
        self.normal_texture = normal_texture
        self.occlusion_texture = occlusion_texture
        self.emissive_texture = emissive_texture
        self.emissive_factor = emissive_factor
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff
        self.double_sided = double_sided
        self.smooth = smooth
        self.wireframe = wireframe

        self._is_transparent = None
        self._tex_flags = None

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
        self._tex_flags = None

    @property
    def occlusion_texture(self):
        return self._occlusion_texture

    @occlusion_texture.setter
    def occlusion_texture(self, value):
        self._occlusion_texture = self._format_texture(value, 'R')
        self._tex_flags = None

    @property
    def emissive_texture(self):
        return self._emissive_texture

    @emissive_texture.setter
    def emissive_texture(self, value):
        self._emissive_texture = self._format_texture(value, 'RGB')
        self._tex_flags = None

    @property
    def emissive_factor(self):
        return self._emissive_factor

    @emissive_factor.setter
    def emissive_factor(self, value):
        self._emissive_factor = format_color_vector(value, 3)

    @property
    def alpha_mode(self):
        return self._alpha_mode

    @alpha_mode.setter
    def alpha_mode(self, value):
        if value not in set(['OPAQUE', 'MASK', 'BLEND']):
            raise ValueError('Invalid alpha mode {}'.format(value))
        self._alpha_mode = alpha_mode
        self._is_transparent = None

    @property
    def alpha_cutoff(self):
        return self._alpha_cutoff

    @alpha_cutoff.setter
    def alpha_cutoff(self, value):
        if value < 0 or value > 1:
            raise ValueError('Alpha cutoff must be in range [0,1]')
        self._alpha_cutoff = float(value)
        self._is_transparent = None

    @property
    def double_sided(self):
        return self._double_sided

    @double_sided.setter
    def double_sided(self, value):
        if not isinstance(value, bool):
            raise TypeError('Double sided must be a boolean value')
        self._double_sided = double_sided

    @property
    def smooth(self):
        return self._smooth

    @smooth.setter
    def smooth(self, value):
        if not isinstance(value, bool):
            raise TypeError('Double sided must be a boolean value')
        self._smooth = smooth

    @property
    def wireframe(self):
        return self._wireframe

    @wireframe.setter
    def wireframe(self, value):
        if not isinstance(value, bool):
            raise TypeError('Wireframe must be a boolean value')
        self._wireframe = wireframe

    @property
    def is_transparent(self):
        if self._is_transparent is None:
            self._is_transparent = self._compute_transparency()
        return self._is_transparent

    @property
    def requires_tangents(self):
        return self.normal_texture is not None

    @property
    def tex_flags(self):
        if self._tex_flags is None:
            self._tex_flags = self._compute_tex_flags()
        return self._tex_flags

    def _compute_transparency(self):
        return False

    def _compute_tex_flags(self):
        tex_flags = TexFlags.NONE
        if self.normal_texture is not None:
            tex_flags |= TexFlags.NORMAL
        if self.occlusion_texture is not None:
            tex_flags |= TexFlags.OCCLUSION
        if self.emissive_texture is not None:
            tex_flags |= TexFlags.EMISSIVE
        return tex_flags

    def _format_texture(self, texture, target_channels='RGB'):
        """Format a texture as a float32 np array.
        """
        if isinstance(texture, Texture):
            return texture
        else:
            source = format_texture_source(texture, target_channels)
            return Texture(source=source, source_channels=target_channels)

class MetallicRoughnessMaterial(Material):
    """Base for standard glTF 2.0 materials.

    Attributes
    ----------
    name : str, optional
        The user-defined name of this object.
    normal_texture : (n,n,3) float, optional
        A tangent-space normal map. The texture contains RGB components in linear space.
        Red maps X to [-1,1], green maps Y to [-1, 1] and blue maps Z to [-1, 1].
    occlusion_texture : (n,n,1) float, optional
        The occlusion texture map. If channel size is >1, occlusion values are sampled
        from the R channel.
    emissive_texture : (n,n,3) float, optional
        The emissive lighting map. Colors should be specified in sRGB space.
    emissive_factor : (3,) float
        The RGB components of the emissive color of the material in linear space.
    alpha_mode : str, optional
        One of 'OPAQUE', 'MASK', or 'BLEND'.
    alpha_cutoff : float, optional
        Specifies cutoff threshold when in MASK mode.
    double_sided : bool, optional
        If True, the material is double sided. If False, back-face culling is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal per vertex.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    base_color_factor : (4,) float
        The RGBA components of the base color of the material. If a texture is specified,
        this factor is multiplied componentwise by the texture. These values are linear.
    base_color_texture : (n,n,4) float
        The base color texture in linear RGB(A) values.
    metallic_factor : float
        The metalness of the material in [0,1]. This value is linear.
    roughness_factor : float
        The roughness of the material in [0,1]. This value is linear.
    metallic_roughness_texture : (n,n,2)
        The metallic-roughness texture. If the number of channels is >2, the metalness
        values are sampled from the B channel and the roughness form the G channel.
        These values are linear.
    """

    def __init__(self,
                 name=None,
                 normal_texture=None,
                 occlusion_texture=None,
                 emissive_texture=None,
                 emissive_factor=None,
                 alpha_mode=None,
                 alpha_cutoff=None,
                 double_sided=False,
                 smooth=True,
                 wireframe=False,
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
            double_sided=double_sided,
            smooth=smooth,
            wireframe=wireframe
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
        self._base_color_factor = format_color_vector(value, 4)
        self._is_transparent = None

    @property
    def base_color_texture(self):
        return self._base_color_texture

    @base_color_texture.setter
    def base_color_texture(self, value):
        self._base_color_texture = self._format_texture(value, 'RGBA')
        self._is_transparent = None
        self._tex_flags = None

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
            raise ValueError('Roughness factor must be in range [0,1]')
        self._roughness_factor = float(value)

    @property
    def metallic_roughness_texture(self):
        return self._metallic_roughness_texture

    @metallic_roughness_texture.setter
    def metallic_roughness_texture(self, value):
        self._metallic_roughness_texture = self._format_texture(value, 'GB')
        self._tex_flags = None

    def _compute_tex_flags(self):
        tex_flags = super(MetallicRoughnessMaterial, self)._compute_tex_flags()
        if self.base_color_texture is not None:
            tex_flags |= TexFlags.BASE_COLOR
        if self.metallic_roughness_texture is not None
            tex_flags |= TexFlags.METALLIC_ROUGHNESS
        return tex_flags

    def _compute_transparency(self):
        if self.alpha_mode == 'OPAQUE':
            return False
        cutoff = self.alpha_cutoff
        if self.alpha_mode == 'BLEND':
            cutoff = 1.0
        if self.base_color_factor[3] < cutoff:
            return True
        if np.any(self.base_color_factor < cutoff):
            return True
        return False
