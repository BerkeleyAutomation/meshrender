import numpy as np

class MaterialProperties(object):
    """A set of material properties describing how an object will look.
    """

    def __init__(self, color=np.array([0.5, 0.5, 0.5]),
                 k_a=1.0, k_d=1.0, k_s = 1.0, alpha=1.0,
                 smooth=False, wireframe=False):
        """Initialize a set of material properties.

        Parameters
        ----------
        color : (3,) float
            The RGB color of the object in (0,1).
        k_a : float
            A multiplier for ambient lighting.
        k_d : float
            A multiplier for diffuse lighting.
        k_s : float
            A multiplier for specular lighting.
        alpha : float
            A multiplier for shininess (higher values indicate
            more reflectivity and smaller highlights).
        smooth : bool
            If True, normals will be interpolated to smooth the mesh.
        wireframe : bool
            If True, the mesh will be rendered as a wireframe.
        """
        self._color = color
        self._k_a = k_a
        self._k_d = k_d
        self._k_s = k_s 
        self._alpha = alpha
        self._smooth = smooth
        self._wireframe = wireframe

    @property
    def color(self):
        """(3,) float: The RGB color of the object in (0,1).
        """
        return self._color

    @property
    def k_a(self):
        """float: A multiplier for ambient lighting.
        """
        return self._k_a

    @property
    def k_d(self):
        """float: A multiplier for diffuse lighting.
        """
        return self._k_d

    @property
    def k_s(self):
        """float: A multiplier for specular lighting. 
        """
        return self._k_s

    @property
    def alpha(self):
        """float: A multiplier for shininess.
        """
        return self._alpha

    @property
    def smooth(self):
        """bool: If True, indicates a smooth rather than piecewise planar surface.
        """
        return self._smooth

    @property
    def wireframe(self):
        """bool: If True, the mesh will be rendered as a wireframe.
        """
        return self._wireframe
