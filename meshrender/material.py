"""Material properties, conforming to the glTF 2.0 standard.
https://git.io/fhkPZ
"""
import abc
import numpy as np
import six

from .constants import TexFlags
from .utils import format_color_vector, format_texture_source
from .texture import Texture

@six.add_metaclass(abc.ABCMeta)
class Material(object):
    """Base for standard glTF 2.0 materials.

    Attributes
    ----------
    name : str, optional
        The user-defined name of this object.
    normalTexture : (n,n,3) float, optional
        A tangent-space normal map. The texture contains RGB components in linear space.
        Red maps X to [-1,1], green maps Y to [-1, 1] and blue maps Z to [-1, 1].
    occlusionTexture : (n,n,1) float, optional
        The occlusion texture map. If channel size is >1, occlusion values are sampled
        from the R channel.
    emissiveTexture : (n,n,3) float, optional
        The emissive lighting map. Colors should be specified in sRGB space.
    emissiveFactor : (3,) float
        The RGB components of the emissive color of the material in linear space.
    alphaMode : str, optional
        One of 'OPAQUE', 'MASK', or 'BLEND'.
    alphaCutoff : float, optional
        Specifies cutoff threshold when in MASK mode.
    doubleSided : bool, optional
        If True, the material is double sided. If False, back-face culling is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal per vertex.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    """

    def __init__(self,
                 name=None,
                 normalTexture=None,
                 occlusionTexture=None,
                 emissiveTexture=None,
                 emissiveFactor=None,
                 alphaMode=None,
                 alphaCutoff=None,
                 doubleSided=False,
                 smooth=True,
                 wireframe=False):

        # Set defaults
        if alphaMode is None:
            alphaMode = 'OPAQUE'

        if alphaCutoff is None:
            alphaCutoff = 0.5

        if emissiveFactor is None:
            emissiveFactor = np.zeros(3).astype(np.float32)

        self.name = name
        self.normalTexture = normalTexture
        self.occlusionTexture = occlusionTexture
        self.emissiveTexture = emissiveTexture
        self.emissiveFactor = emissiveFactor
        self.alphaMode = alphaMode
        self.alphaCutoff = alphaCutoff
        self.doubleSided = doubleSided
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
    def normalTexture(self):
        return self._normalTexture

    @normalTexture.setter
    def normalTexture(self, value):
        # TODO TMP
        self._normalTexture = self._format_texture(value, 'RGB')
        self._tex_flags = None

    @property
    def occlusionTexture(self):
        return self._occlusionTexture

    @occlusionTexture.setter
    def occlusionTexture(self, value):
        self._occlusionTexture = self._format_texture(value, 'R')
        self._tex_flags = None

    @property
    def emissiveTexture(self):
        return self._emissiveTexture

    @emissiveTexture.setter
    def emissiveTexture(self, value):
        self._emissiveTexture = self._format_texture(value, 'RGB')
        self._tex_flags = None

    @property
    def emissiveFactor(self):
        return self._emissiveFactor

    @emissiveFactor.setter
    def emissiveFactor(self, value):
        if value is None:
            value = np.zeros(3)
        self._emissiveFactor = format_color_vector(value, 3)

    @property
    def alphaMode(self):
        return self._alphaMode

    @alphaMode.setter
    def alphaMode(self, value):
        if value not in set(['OPAQUE', 'MASK', 'BLEND']):
            raise ValueError('Invalid alpha mode {}'.format(value))
        self._alphaMode = value
        self._is_transparent = None

    @property
    def alphaCutoff(self):
        return self._alphaCutoff

    @alphaCutoff.setter
    def alphaCutoff(self, value):
        if value < 0 or value > 1:
            raise ValueError('Alpha cutoff must be in range [0,1]')
        self._alphaCutoff = float(value)
        self._is_transparent = None

    @property
    def doubleSided(self):
        return self._doubleSided

    @doubleSided.setter
    def doubleSided(self, value):
        if not isinstance(value, bool):
            raise TypeError('Double sided must be a boolean value')
        self._doubleSided = value

    @property
    def smooth(self):
        return self._smooth

    @smooth.setter
    def smooth(self, value):
        if not isinstance(value, bool):
            raise TypeError('Double sided must be a boolean value')
        self._smooth = value

    @property
    def wireframe(self):
        return self._wireframe

    @wireframe.setter
    def wireframe(self, value):
        if not isinstance(value, bool):
            raise TypeError('Wireframe must be a boolean value')
        self._wireframe = value

    @property
    def is_transparent(self):
        if self._is_transparent is None:
            self._is_transparent = self._compute_transparency()
        return self._is_transparent

    @property
    def requires_tangents(self):
        return self.normalTexture is not None

    @property
    def tex_flags(self):
        if self._tex_flags is None:
            self._tex_flags = self._compute_tex_flags()
        return self._tex_flags

    @property
    def textures(self):
        return self._compute_textures()

    def _compute_transparency(self):
        return False

    def _compute_tex_flags(self):
        tex_flags = TexFlags.NONE
        if self.normalTexture is not None:
            tex_flags |= TexFlags.NORMAL
        if self.occlusionTexture is not None:
            tex_flags |= TexFlags.OCCLUSION
        if self.emissiveTexture is not None:
            tex_flags |= TexFlags.EMISSIVE
        return tex_flags

    def _compute_textures(self):
        all_textures = [self.normalTexture, self.occlusionTexture, self.emissiveTexture]
        textures = set([t for t in all_textures if t is not None])
        return textures

    def _format_texture(self, texture, target_channels='RGB'):
        """Format a texture as a float32 np array.
        """
        if isinstance(texture, Texture) or texture is None:
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
    normalTexture : (n,n,3) float, optional
        A tangent-space normal map. The texture contains RGB components in linear space.
        Red maps X to [-1,1], green maps Y to [-1, 1] and blue maps Z to [-1, 1].
    occlusionTexture : (n,n,1) float, optional
        The occlusion texture map. If channel size is >1, occlusion values are sampled
        from the R channel.
    emissiveTexture : (n,n,3) float, optional
        The emissive lighting map. Colors should be specified in sRGB space.
    emissiveFactor : (3,) float
        The RGB components of the emissive color of the material in linear space.
    alphaMode : str, optional
        One of 'OPAQUE', 'MASK', or 'BLEND'.
    alphaCutoff : float, optional
        Specifies cutoff threshold when in MASK mode.
    doubleSided : bool, optional
        If True, the material is double sided. If False, back-face culling is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal per vertex.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    baseColorFactor : (4,) float
        The RGBA components of the base color of the material. If a texture is specified,
        this factor is multiplied componentwise by the texture. These values are linear.
    baseColorTexture : (n,n,4) float
        The base color texture in linear RGB(A) values.
    metallicFactor : float
        The metalness of the material in [0,1]. This value is linear.
    roughnessFactor : float
        The roughness of the material in [0,1]. This value is linear.
    metallicRoughnessTexture : (n,n,2)
        The metallic-roughness texture. If the number of channels is >2, the metalness
        values are sampled from the B channel and the roughness form the G channel.
        These values are linear.
    """

    def __init__(self,
                 name=None,
                 normalTexture=None,
                 occlusionTexture=None,
                 emissiveTexture=None,
                 emissiveFactor=None,
                 alphaMode=None,
                 alphaCutoff=None,
                 doubleSided=False,
                 smooth=True,
                 wireframe=False,
                 baseColorFactor=None,
                 baseColorTexture=None,
                 metallicFactor=1.0,
                 roughnessFactor=1.0,
                 metallicRoughnessTexture=None):
        super(MetallicRoughnessMaterial, self).__init__(
            name=name,
            normalTexture=normalTexture,
            occlusionTexture=occlusionTexture,
            emissiveTexture=emissiveTexture,
            emissiveFactor=emissiveFactor,
            alphaMode=alphaMode,
            alphaCutoff=alphaCutoff,
            doubleSided=doubleSided,
            smooth=smooth,
            wireframe=wireframe
        )

        # Set defaults
        if baseColorFactor is None:
            baseColorFactor = np.ones(4).astype(np.float32)

        self.baseColorFactor = baseColorFactor
        self.baseColorTexture = baseColorTexture
        self.metallicFactor = metallicFactor
        self.roughnessFactor = roughnessFactor
        self.metallicRoughnessTexture = metallicRoughnessTexture

    @property
    def baseColorFactor(self):
        return self._baseColorFactor

    @baseColorFactor.setter
    def baseColorFactor(self, value):
        if value is None:
            value = np.ones(4)
        self._baseColorFactor = format_color_vector(value, 4)
        self._is_transparent = None

    @property
    def baseColorTexture(self):
        return self._baseColorTexture

    @baseColorTexture.setter
    def baseColorTexture(self, value):
        self._baseColorTexture = self._format_texture(value, 'RGBA')
        self._is_transparent = None
        self._tex_flags = None

    @property
    def metallicFactor(self):
        return self._metallicFactor

    @metallicFactor.setter
    def metallicFactor(self, value):
        if value is None:
            value = 1.0
        if value < 0 or value > 1:
            raise ValueError('Metallic factor must be in range [0,1]')
        self._metallicFactor = float(value)

    @property
    def roughnessFactor(self):
        return self.RoughnessFactor

    @roughnessFactor.setter
    def roughnessFactor(self, value):
        if value is None:
            value = 1.0
        if value < 0 or value > 1:
            raise ValueError('Roughness factor must be in range [0,1]')
        self.RoughnessFactor = float(value)

    @property
    def metallicRoughnessTexture(self):
        return self._metallicRoughnessTexture

    @metallicRoughnessTexture.setter
    def metallicRoughnessTexture(self, value):
        self._metallicRoughnessTexture = self._format_texture(value, 'GB')
        self._tex_flags = None

    def _compute_tex_flags(self):
        tex_flags = super(MetallicRoughnessMaterial, self)._compute_tex_flags()
        if self.baseColorTexture is not None:
            tex_flags |= TexFlags.BASE_COLOR
        if self.metallicRoughnessTexture is not None:
            tex_flags |= TexFlags.METALLIC_ROUGHNESS
        return tex_flags

    def _compute_transparency(self):
        if self.alphaMode == 'OPAQUE':
            return False
        cutoff = self.alphaCutoff
        if self.alphaMode == 'BLEND':
            cutoff = 1.0
        if self.baseColorFactor[3] < cutoff:
            return True
        if np.any(self.baseColorFactor < cutoff):
            return True
        return False

    def _compute_textures(self):
        textures = super(MetallicRoughnessMaterial, self)._compute_textures()
        all_textures = [self.baseColorTexture, self.metallicRoughnessTexture]
        all_textures = {t for t in all_textures if t is not None}
        textures |= all_textures
        return textures


class SpecularGlossinessMaterial(Material):
    """Base for standard glTF 2.0 materials.

    Attributes
    ----------
    name : str, optional
        The user-defined name of this object.
    normalTexture : (n,n,3) float, optional
        A tangent-space normal map. The texture contains RGB components in linear space.
        Red maps X to [-1,1], green maps Y to [-1, 1] and blue maps Z to [-1, 1].
    occlusionTexture : (n,n,1) float, optional
        The occlusion texture map. If channel size is >1, occlusion values are sampled
        from the R channel.
    emissiveTexture : (n,n,3) float, optional
        The emissive lighting map. Colors should be specified in sRGB space.
    emissiveFactor : (3,) float
        The RGB components of the emissive color of the material in linear space.
    alphaMode : str, optional
        One of 'OPAQUE', 'MASK', or 'BLEND'.
    alphaCutoff : float, optional
        Specifies cutoff threshold when in MASK mode.
    doubleSided : bool, optional
        If True, the material is double sided. If False, back-face culling is enabled.
    smooth : bool, optional
        If True, the material is rendered smoothly by using only one normal per vertex.
    wireframe : bool, optional
        If True, the material is rendered in wireframe mode.
    diffuseFactor : (4,) float
        The RGBA components of the diffuse of the material. If a texture is specified,
        this factor is multiplied componentwise by the texture. These values are linear.
    diffuseTexture : (n,n,4) float
        The diffuse texture in linear RGB(A) values.
    specularFactor : (3,) float
        The specular color of the material in [0,1]. This value is linear.
    glossinessFactor : float
        The glossiness of the material in [0,1]. This value is linear.
    specularGlossinessTexture : (n,n,4)
        The specular-glossiness texture. The specular color is in sRGB space in the RGB
        channel and the glossiness value is in the A channel in linear space.
    """

    def __init__(self,
                 name=None,
                 normalTexture=None,
                 occlusionTexture=None,
                 emissiveTexture=None,
                 emissiveFactor=None,
                 alphaMode=None,
                 alphaCutoff=None,
                 doubleSided=False,
                 smooth=True,
                 wireframe=False,
                 diffuseFactor=None,
                 diffuseTexture=None,
                 specularFactor=None,
                 glossinessFactor=1.0,
                 specularGlossinessTexture=None):
        super(SpecularGlossinessMaterial, self).__init__(
            name=name,
            normalTexture=normalTexture,
            occlusionTexture=occlusionTexture,
            emissiveTexture=emissiveTexture,
            emissiveFactor=emissiveFactor,
            alphaMode=alphaMode,
            alphaCutoff=alphaCutoff,
            doubleSided=doubleSided,
            smooth=smooth,
            wireframe=wireframe
        )

        # Set defaults
        if diffuseFactor is None:
            diffuseFactor = np.ones(4).astype(np.float32)
        if specularFactor is None:
            specularFactor = np.ones(3).astype(np.float32)

        self.diffuseFactor = diffuseFactor
        self.diffuseTexture = diffuseTexture
        self.specularFactor = specularFactor
        self.glossinessFactor = glossinessFactor
        self.specularGlossinessTexture = specularGlossinessTexture

    @property
    def diffuseFactor(self):
        return self._diffuseFactor

    @diffuseFactor.setter
    def diffuseFactor(self, value):
        self._diffuseFactor = format_color_vector(value, 4)
        self._is_transparent = None

    @property
    def diffuseTexture(self):
        return self._diffuseTexture

    @diffuseTexture.setter
    def diffuseTexture(self, value):
        self._diffuseTexture = self._format_texture(value, 'RGBA')
        self._is_transparent = None
        self._tex_flags = None

    @property
    def specularFactor(self):
        return self._specularFactor

    @specularFactor.setter
    def specularFactor(self, value):
        self._specularFactor = format_color_vector(value, 3)

    @property
    def glossinessFactor(self):
        return self.glossinessFactor

    @glossinessFactor.setter
    def glossinessFactor(self, value):
        if value < 0 or value > 1:
            raise ValueError('glossiness factor must be in range [0,1]')
        self._glossinessFactor = float(value)

    @property
    def specularGlossinessTexture(self):
        return self._specularGlossinessTexture

    @specularGlossinessTexture.setter
    def specularGlossinessTexture(self, value):
        self._specularGlossinessTexture = self._format_texture(value, 'GB')
        self._tex_flags = None

    def _compute_tex_flags(self):
        tex_flags = super(SpecularGlossinessMaterial, self)._compute_tex_flags()
        if self.diffuseTexture is not None:
            tex_flags |= TexFlags.DIFFUSE
        if self.specularGlossinessTexture is not None:
            tex_flags |= TexFlags.SPECULAR_GLOSSINESS
        return tex_flags

    def _compute_transparency(self):
        if self.alphaMode == 'OPAQUE':
            return False
        cutoff = self.alphaCutoff
        if self.alphaMode == 'BLEND':
            cutoff = 1.0
        if self.diffuseFactor[3] < cutoff:
            return True
        if np.any(self.diffuseFactor < cutoff):
            return True
        return False

    def _compute_textures(self):
        textures = super(SpecularGlossinessMaterial, self)._compute_textures()
        all_textures = [self.diffuseTexture, self.specularGlossinessTexture]
        all_textures = {t for t in all_textures if t is not None}
        textures |= all_textures
        return textures
