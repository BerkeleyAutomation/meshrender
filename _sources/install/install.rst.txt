Python Installation
~~~~~~~~~~~~~~~~~~~

1. Clone the repository
"""""""""""""""""""""""
Clone or download our source code from `Github`_. ::

    $ git clone https://github.com/BerkeleyAutomation/meshrender.git

.. _Github: https://github.com/BerkeleyAutomation/meshrender

2. Run installation script
""""""""""""""""""""""""""
Change directories into the `meshrender` repository and run ::

    $ python setup.py install

This will install `meshrender` in your current Python environment.

Dependencies
~~~~~~~~~~~~
The `meshrender` module depends on a few external modules. `numpy`, `scipy`,
`PyOpenGL`, `PyOpenGL_accelerate`, `trimesh`, and `ctypes` are all pip installable and
should be installed automatically by the installation script.

However, `autolab_core`_ and `perception`_ are Berkeley AUTOLab-specific and aren't yet on PyPi.
Install them from Github using their documetation.

.. _autolab_core: http://www.github.com/BerkeleyAutomation/autolab_core
.. _perception: http://www.github.com/BerkeleyAutomation/perception

Documentation
~~~~~~~~~~~~~

Building
""""""""
Building `meshrender`'s documentation requires a few extra dependencies --
specifically, `sphinx`_ and a few plugins.

.. _sphinx: http://www.sphinx-doc.org/en/1.4.8/

To install the dependencies required, simply run ::

    $ pip install -r docs_requirements.txt

Then, go to the `docs` directory and run ``make`` with the appropriate target.
For example, ::

    $ cd docs/
    $ make html

will generate a set of web pages. Any documentation files
generated in this manner can be found in `docs/build`.

Deploying
"""""""""
To deploy documentation to the Github Pages site for the repository,
simply push any changes to the documentation source to master
and then run ::

    $ . gh_deploy.sh

from the `docs` folder. This script will automatically checkout the
``gh-pages`` branch, build the documentation from source, and push it
to Github.

