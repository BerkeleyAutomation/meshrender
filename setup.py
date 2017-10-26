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
    'trimesh',
    'PyOpenGL',
    'PyOpenGL_accelerate',
    #'autolab_core',
    #'perception',
]

setup(
    name='meshrender',
    version=__version__,
    description='Python utilities for rendering scenes containing 3D meshes',
    long_description='A set of Python utilities for rendering 3D scenes, based on PyOpenGL and target at OpenGL 3+.',
    url='https://github.com/BerkeleyAutomation/meshrender',
    author='Matthew Matl',
    author_email='mmatl@eecs.berkeley.edu',
    license = "Apache",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='rendering opengl',
    packages=['meshrender'],
    install_requires=requirements,
)
