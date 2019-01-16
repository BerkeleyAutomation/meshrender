"""
Setup of meshrender Python codebase.

Author: Matthew Matl
"""

from setuptools import setup

# load __version__
exec(open('meshrender/version.py').read())

requirements = [
    'freetype-py',
    'imageio',                      # For Image I/O
    'numpy',                        # Numpy
    'Pillow',                       # For Trimesh texture conversions
    'pyglet==1.4.0a1',              # For the pyglet viewer
    'PyOpenGL==3.1.0',              # For OpenGL
    'PyOpenGL_accelerate==3.1.0',   # For OpenGL
    'six',                          # For Python 2/3 interop
    'trimesh',                      # For meshes
]

setup(
    name = 'meshrender',
    version = __version__,
    description = 'Python utilities for 3D PBR rendering and visualization.',
    long_description = 'A set of Python utilities for easy physically-based rendering (PBR) of scenes. Compliant with the glTF 2.0 standard.',
    author = 'Matthew Matl',
    author_email = 'matthewcmatl@gmail.com',
    license = "Apache Software License",
    url = 'https://github.com/BerkeleyAutomation/meshrender',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords = 'rendering opengl 3d visualization pbr gltf',
    packages = ['meshrender'],
    setup_requires = requirements,
    install_requires = requirements,
    extras_require = { 'docs' : [
            'sphinx',
            'sphinxcontrib-napoleon',
            'sphinx_rtd_theme'
        ],
    }
)
