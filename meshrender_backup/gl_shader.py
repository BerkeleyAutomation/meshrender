import abc
import six

from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shader_utils

class Program(object):

    def __init__(self, shader_filenames):

        shader_ids = []

        # Load and compile shaders
        for shader_filename in shader_filenames:
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
            shaders_ids.append(shader_id)

        # Compile program
        self.program_id = gl_shader_utils.compileProgram(shader_ids)

        # Free shaders
        for sid in shader_ids:
            glDeleteShader(sid)

    def bind(self):
        glUseProgram(self.program_id)

    def unbind(self):
        glUseProgram(0)

    def delete(self):
        if self.program_id is not None:
            glDeleteProgram(self.program_id)
            self.program_id = None

    def __del__(self):
        self.delete()

    def set_uniform(self, name, value, unsigned=False):
        loc = glGetUniformLocation(self.program_id, name)

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
            if value.dtype in set([np.uint8, np.uint16, np.uint32, np.uint64]) or unsigned:
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
            elif value.dtype in set([np.int8, np.int16, np.int32, np.int64]):
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
            elif value.dtype in set([np.float16, np.float32, np.float64]):
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
                    if value.shape == (3,2):
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
