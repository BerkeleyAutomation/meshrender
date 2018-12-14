class Material(object):
    """A set of material properties for a mesh.

    Attributes
    ----------
    diffuse : (3,) float or (n,n,d) float
        Either a single RGB diffuse color or a texture map image.
        Final dimension of texture map can be 1,3, or 4.
    specular : (3,) float or (n,n,d) float
        Either a single RGB specular color or a texture map image.
        Final dimension of texture map can be 1,3, or 4.
    shininess : float
        Exponential shininess term.
    emission : (3,) float or (n,n,d) float
        Either a single RGB emission color or a texture map image.
        Final dimension of texture map can be 1,3, or 4.
    normal : (n,n,3) float
        Normal map image.
    smooth : bool
        If True, the shape is rendered smoothly.
    wireframe : bool
        If True, the shape is rendered in wireframe mode.
    """

    def __init__(self, diffuse, specular, shininess, emission=None,
                 normal=None, smooth=False, wireframe=False):
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.emission = emission
        self.normal = normal
        self.smooth = smooth
        self.wireframe = wireframe

    def copy(self):
        diffuse = self.diffuse.copy()
        specular = self.specular.copy()
        emission = (None if self.emission is None else self.emission.copy())
        smooth = (None if self.smooth is None else self.smooth.copy())

        return Material(diffuse, specular, self.shininess, emission,
                        normal, self.smooth, self.wireframe)
