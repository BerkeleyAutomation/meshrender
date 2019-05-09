"""
Setup of meshrender Python codebase.
Authors: Matthew Matl and Jeff Mahler
"""

from setuptools import setup

# load __version__
exec(open('meshrender/version.py').read())

requirements = [
    'numpy',
    'scipy',
    'trimesh[easy]',
    'PyOpenGL>=3.1.0',
    'pyglet>=1.4.0b1',
    'imageio',
    'autolab_core',
    'autolab_perception'
]

setup(
    name = 'meshrender',
    version = __version__,
    description = 'Python utilities for rendering scenes containing 3D meshes',
    long_description = 'A set of Python utilities for rendering 3D scenes, based on PyOpenGL and target at OpenGL 3+.',
    author = 'Matthew Matl',
    author_email = 'mmatl@eecs.berkeley.edu',
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
    keywords = 'rendering opengl 3d visualization',
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
