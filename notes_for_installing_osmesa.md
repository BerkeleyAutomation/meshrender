# Guide for using OSMesa as an offscreen rendering backend
If you want to render offscreen without hijacking a local display,
you can use the software-based OSMesa renderer by following these
instructions.

## Install the dependencies:
```sh
sudo apt-get install llvm-6
sudo apt-get install freeglut3 freeglut3-dev
```

## Installation
Download the current release at: <ftp://ftp.freedesktop.org/pub/mesa/>.
Unpack and go to the source folder:
```sh
tar zxf mesa-*.*.*.tar.gz
cd mesa-*
```
Replace `PREFIX` with the path you want to install Mesa at.
Make sure we do not install Mesa into the system path.
Adapt the `llvm-config-x.x` to your own machine's llvm (e.g. `llvm-config-6.0`
        if you installed `llvm` with the above command).

Configure the install to use the `gallium` offscreen renderer, which supports
modern OpenGL.
```sh
./configure                                         \
  --prefix=PREFIX                                   \
  --enable-opengl --disable-gles1 --disable-gles2   \
  --disable-va --disable-xvmc --disable-vdpau       \
  --enable-shared-glapi                             \
  --disable-texture-float                           \
  --enable-gallium-llvm --enable-llvm-shared-libs   \
  --with-gallium-drivers=swrast,swr                 \
  --disable-dri --with-dri-drivers=                 \
  --disable-egl --with-egl-platforms= --disable-gbm \
  --disable-glx                                     \
  --disable-osmesa --enable-gallium-osmesa          \
  ac_cv_path_LLVM_CONFIG=llvm-config-x.x
make -j8
make install
```

Add the following lines to your `~/.bashrc` file and change `MESA_HOME` to your mesa installation path:
```sh
# Mesa
MESA_HOME=/path/to/your/mesa/installation
export LIBRARY_PATH=$LIBRARY_PATH:$MESA_HOME/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MESA_HOME/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$MESA_HOME/include/
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$MESA_HOME/include/
```

Finally, use my fork of PyOpenGL (at least until the main release has
integrated my patch that supports getting modern contexts):

```
git clone git@github.com:mmatl/PyOpenGL.git
cd PyOpenGL
python setup.py install
```

## Usage
Before running any script using the `OffscreenRenderer`, make sure to set the
`PYOPENGL_PLATFORM` environment variable to `osmesa`. For example:

```
PYOPENGL_PLATFORM=osmesa python run_rendering_script.py
```

If you do this, you won't be able to use the `Viewer`, but you will be able do
do offscreen rendering without a display and even over SSH.
