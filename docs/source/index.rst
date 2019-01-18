.. core documentation master file, created by
   sphinx-quickstart on Sun Oct 16 14:33:48 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Meshrender Documentation
========================
Meshrender is a Python 2/3 implementation of Physically-Based Rendering (PBR).
If is mostly compliant with the glTF 2.0 specification, and it makes it easy
to render 3D scenes in pure Python. Dependencies are light and all
pip-installable.

.. image:: scene.png

Meshrender supports rendering objects with metallic-roughness textures,
normal maps, ambient occlusion textures, emission textures, and shadows.

.. image:: damaged_helmet.png

.. toctree::
   :maxdepth: 2
   :caption: Installation Guide

   install/install.rst

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   :glob:

   api/*

.. toctree::
    :maxdepth: 2
    :caption: Examples

    examples/example.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

