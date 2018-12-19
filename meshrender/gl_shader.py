import abc
import six

from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders

class Program(object):

    def __init__(self, shaders):
        self.shaders = shaders
        self.program = shaders.compileProgram(*shaders)

    def set_uniform(self, name, value, unsigned=False):
        loc = glGetUniformLocation(self.program, name)

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

@six.add_metaclass(abc.ABCMeta)
class Shader(object):

    def __init__(self, name, shader_filename):
        self.name = name
        shader_str = Shader._read_file(shader_filename)
        self.shader = gl_shaders.compileShader(shader_str, self.shader_type())

    @abstractmethod
    def shader_type(self):
        pass

    @staticmethod
    def _read_file(file_obj):
        if hasattr(file_obj, 'read'):
            return file_obj.read()
        else:
            data = None
            with f as open(file_obj):
                data = f.read()
            return data


class VertexShader(OpenGLShader):

    def shader_type(self):
        return GL_VERTEX_SHADER

class FragmentShader(OpenGLShader):

    def shader_type(self):
        return GL_VERTEX_SHADER

class GeometryShader(OpenGLShader):

    def shader_type(self):
        return GL_VERTEX_SHADER
