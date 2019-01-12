"""OpenGL shader program wrapper.
"""
import numpy as np
import os
import re

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

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        """Get a program via a list of shader files to include in the program.

        Parameters
        ----------
        shader_filenames : list of str
            Path to shader files that should be compiled into this program.
            Acceptable file extensions are .vert, .geom, and .frag.
        """
        shader_names = []
        if defines is None:
            defines = []
        shader_filenames = [x for x in [vertex_shader, fragment_shader, geometry_shader] if x is not None]
        for fn in shader_filenames:
            if fn is None:
                continue
            _, name = os.path.split(fn)
            shader_names.append(name)
        key = tuple(sorted(shader_names + defines))

        if key not in self._program_cache:
            shader_filenames = [os.path.join(self.shader_dir, fn) for fn in shader_filenames]
            if len(shader_filenames) == 2:
                shader_filenames.append(None)
            vs, fs, gs = shader_filenames
            self._program_cache[key] = ShaderProgram(vertex_shader=vs, fragment_shader=fs,
                                                     geometry_shader=gs, defines=defines)
        return self._program_cache[key]

    def clear(self):
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

    def __init__(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):

        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        self.geometry_shader = geometry_shader

        if defines is None:
            self.defines = set()
        else:
            self.defines = set(defines)
        self._program_id = None

    def _add_to_context(self):
        if self._program_id is not None:
            raise ValueError('Shader program already in context')
        shader_ids = []

        # Load vert shader
        shader_ids.append(gl_shader_utils.compileShader(
            self._load(self.vertex_shader), GL_VERTEX_SHADER)
        )
        # Load frag shader
        shader_ids.append(gl_shader_utils.compileShader(
            self._load(self.fragment_shader), GL_FRAGMENT_SHADER)
        )
        # Load geometry shader
        if self.geometry_shader is not None:
            shader_ids.append(gl_shader_utils.compileShader(
                self._load(self.geometry_shader), GL_GEOMETRY_SHADER)
            )

        # Compile program
        self._program_id = gl_shader_utils.compileProgram(*shader_ids)

        # Free shaders
        for sid in shader_ids:
            glDeleteShader(sid)

    def _in_context(self):
        return self._program_id is not None

    def _remove_from_context(self):
        if self._program_id is not None:
            glDeleteProgram(self._program_id)
            self._program_id = None

    def _load(self, shader_filename):
        path, _ = os.path.split(shader_filename)
        #import_re = re.compile('^(.*)#import\s+(.*)\s+$', re.MULTILINE)

        #def recursive_load(matchobj, path):
        #    indent = matchobj.group(1)
        #    fname = os.path.join(path, matchobj.group(2))
        #    new_path, _ = os.path.split(fname)
        #    new_path = os.path.realpath(new_path)
        #    with open(fname) as f:
        #        text = f.read()
        #    text = indent + text
        #    text = text.replace('\n', '\n{}'.format(indent), text.count('\n') - 1)
        #    return re.sub(import_re, lambda m : recursive_load(m, new_path), text)

        #text = re.sub(import_re, lambda m : recursive_load(m, path), text)
        with open(shader_filename) as f:
            text = f.read()

        def defined(matchobj):
            if matchobj.group(1) in self.defines:
                return '#if 1'
            else:
                return '#if 0'

        def ndefined(matchobj):
            if matchobj.group(1) in self.defines:
                return '#if 0'
            else:
                return '#if 1'


        def_regex = re.compile('#ifdef\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*$', re.MULTILINE)
        ndef_regex = re.compile('#ifndef\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*$', re.MULTILINE)
        text = re.sub(def_regex, defined, text)
        text = re.sub(ndef_regex, ndefined, text)

        return text

    def _bind(self):
        """Bind this shader program to the current OpenGL context.
        """
        if self._program_id is None:
            raise ValueError('Cannot bind program that is not in context')
        glUseProgram(self._program_id)

    def _unbind(self):
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
        self._remove_from_context()

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
