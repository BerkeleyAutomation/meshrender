"""OpenGL shader program wrapper.
"""
import numpy as np
import os

from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shader_utils

class ShaderProgramCache(object):
    """A cache for shader programs.
    """

    def __init__(self, shader_dir=None):
        self._program_cache = {}
        self.shader_dir = shader_dir
        if self.shader_dir is None:
            base_dir, _ = os.path.split(os.path.realpath(__file__))
            self.shader_dir = os.path.join(base_dir, 'shaders')

    def get_program(self, shader_filenames):
        """Get a program via a list of shader files to include in the program.

        Parameters
        ----------
        shader_filenames : list of str
            Path to shader files that should be compiled into this program.
            Acceptable file extensions are .vert, .geom, and .frag.
        """
        shader_names = []
        for fn in shader_filenames:
            _, name = os.path.split(fn)
            shader_names.append(name)
        shader_names = tuple(sorted(shader_names))

        if shader_names not in self._program_cache:
            shader_filenames = [os.path.join(self.shader_dir, fn) for fn in shader_filenames]
            self._program_cache[shader_names] = ShaderProgram(shader_filenames)
        return self._program_cache[shader_names]

    def clear(self):
        self.delete()

    def delete(self):
        """Delete all cached shader programs.
        """
        for key in self._program_cache:
            self._program_cache[key].delete()
        self._program_cache = {}

class ShaderProgram(object):
    """A thin wrapper about OpenGL shader programs that supports easy creation,
    binding, and uniform-setting.

    Attributes
    ----------
    shader_filenames: list of str
        Path to shader files that should be compiled into this program.
        Acceptable file extensions are .vert, .geom, and .frag.
    """

    def __init__(self, shader_filenames):

        shader_ids = []
        self.shader_filenames = shader_filenames

        # Load and compile shaders
        for shader_filename in shader_filenames:
            # Add proper directory to shader filename
            # Get shader type
            path, base = os.path.split(shader_filename)
            _, ext = os.path.splitext(base)
            if ext == '.vert':
                shader_type = GL_VERTEX_SHADER
            elif ext == '.frag':
                shader_type = GL_FRAGMENT_SHADER
            elif ext == '.geom':
                shader_type = GL_GEOMETRY_SHADER
            else:
                raise ValueError('Unsupported shader file extension: {}'.format(ext))

            # Load shader file
            with open(shader_filename) as f:
                shader_str = f.read()

            shader_id = gl_shader_utils.compileShader(shader_str, shader_type)
            shader_ids.append(shader_id)

        # Compile program
        self._program_id = gl_shader_utils.compileProgram(*shader_ids)

        # Free shaders
        for sid in shader_ids:
            glDeleteShader(sid)

    def bind(self):
        """Bind this shader program to the current OpenGL context.
        """
        glUseProgram(self._program_id)

    def unbind(self):
        """Unbind this shader program from the current OpenGL context.
        """
        glUseProgram(0)

    def _print_uniforms(self):
        print '============================='
        x = glGetProgramiv(self._program_id, GL_ACTIVE_UNIFORMS)
        data = (GLfloat * 16)()
        for i in range(x):
            name, _, _ = glGetActiveUniform(self._program_id, i)
            loc = glGetUniformLocation(self._program_id, name)
            a = glGetUniformfv(self._program_id, loc, data)
            print name, list(data)
        print '-----------------------------'

    def delete(self):
        """Delete this shader program from the current OpenGL context.
        """
        if self._program_id is not None:
            glDeleteProgram(self._program_id)
            self._program_id = None

    def __del__(self):
        if self._program_id is not None:
            self.delete()

    def set_uniform(self, name, value, unsigned=False):
        """Set a uniform value in the current shader program.

        Parameters
        ----------
        name : str
            Name of the uniform to set.
        value : int, float, or ndarray
            Value to set the uniform to.
        unsigned : bool
            If True, ints will be treated as unsigned values.
        """
        loc = glGetUniformLocation(self._program_id, name)

        if loc == -1:
            raise ValueError('Invalid shader name variable: {}'.format(name))

        # Call correct uniform function
        if isinstance(value, float):
            glUniform1f(loc, value)
        elif isinstance(value, int):
            if unsigned:
                glUniform1ui(loc, value)
            else:
                glUniform1i(loc, value)
        elif isinstance(value, bool):
            if unsigned:
                glUniform1ui(loc, int(value))
            else:
                glUniform1i(loc, int(value))
        elif isinstance(value, np.ndarray):
            # Set correct data type
            if np.issubdtype(value.dtype, np.unsignedinteger) or unsigned:
                value = value.astype(np.uint32)
                if value.ndim == 1:
                    if value.shape[0] == 1:
                        glUniform1uiv(loc, 1, value)
                    elif value.shape[0] == 2:
                        glUniform2uiv(loc, 1, value)
                    elif value.shape[0] == 3:
                        glUniform3uiv(loc, 1, value)
                    elif value.shape[0] == 4:
                        glUniform4uiv(loc, 1, value)
                    else:
                        raise ValueError('Invalid data type')
                else:
                    raise ValueError('Invalid data type')
            elif np.issubdtype(value.dtype, np.signedinteger):
                value = value.astype(np.int32)
                if value.ndim == 1:
                    if value.shape[0] == 1:
                        glUniform1iv(loc, 1, value)
                    elif value.shape[0] == 2:
                        glUniform2iv(loc, 1, value)
                    elif value.shape[0] == 3:
                        glUniform3iv(loc, 1, value)
                    elif value.shape[0] == 4:
                        glUniform4iv(loc, 1, value)
                    else:
                        raise ValueError('Invalid data type')
                else:
                    raise ValueError('Invalid data type')
            elif np.issubdtype(value.dtype, np.floating):
                value = value.astype(np.float32)
                if value.ndim == 1:
                    if value.shape[0] == 1:
                        glUniform1fv(loc, 1, value)
                    elif value.shape[0] == 2:
                        glUniform2fv(loc, 1, value)
                    elif value.shape[0] == 3:
                        glUniform3fv(loc, 1, value)
                    elif value.shape[0] == 4:
                        glUniform4fv(loc, 1, value)
                    else:
                        raise ValueError('Invalid data type')
                elif value.ndim == 2:
                    if value.shape == (2,2):
                        glUniformMatrix2fv(loc, 1, GL_TRUE, value)
                    elif value.shape == (2,3):
                        glUniformMatrix2x3fv(loc, 1, GL_TRUE, value)
                    elif value.shape == (2,4):
                        glUniformMatrix2x4fv(loc, 1, GL_TRUE, value)
                    elif value.shape == (3,2):
                        glUniformMatrix3x2fv(loc, 1, GL_TRUE, value)
                    elif value.shape == (3,3):
                        glUniformMatrix3fv(loc, 1, GL_TRUE, value)
                    elif value.shape == (3,4):
                        glUniformMatrix3x4fv(loc, 1, GL_TRUE, value)
                    if value.shape == (4,2):
                        glUniformMatrix4x2fv(loc, 1, GL_TRUE, value)
                    elif value.shape == (4,3):
                        glUniformMatrix4x3fv(loc, 1, GL_TRUE, value)
                    elif value.shape == (4,4):
                        glUniformMatrix4fv(loc, 1, GL_TRUE, value)
                    else:
                        raise ValueError('Invalid data type')
                else:
                    raise ValueError('Invalid data type')
            else:
                raise ValueError('Invalid data type')
        else:
            raise ValueError('Invalid data type')
