from .constants import Shading

from .gl_shader import Program, Shader

class ProgramRegistry(object):

    def __init__(self):
        self._shader_cache = {}
        self._program_cache = {}

    def get_program(self, shading_mode):
        if shading_mode == Shading.DEFAULT:
            return self._load_program(['default.vert', 'default.frag'])
        elif shading_mode == Shading.POINT_CLOUD:
            return self._load_program(['point_cloud.vert', 'point_cloud.frag'])
        elif shading_mode == Shading.FACE_COLORS | shading_mode == SHADING.VERT_COLORS:
            return self._load_program(['vertex_colors.vert', 'vertex_colors.frag'])
        elif shading_mode == Shading.VERT_NORMALS:
            return self._load_program(['vertex_normals.vert', 'vertex_normals.geom',
                                       'vertex_normals.frag'])
        else:
            raise NotImplementedError('Shading mode {} not implemented'.format(shading_mode))

    def _load_program(self, shader_names):
        shader_names = tuple(sorted(shader_names))

        if shader_names in self._program_cache:
            return self._program_cache[shader_names]

        shaders = [self._load_shader(n) for n in shader_names]
        program = Program(shaders)
        self._program_cache[shader_names] = program
        return program

    def _load_shader(self, shader_name):
        if shader_name in self._shader_cache:
            return self.shader_cache[shader_name]

        # Create shader file path
        file_path = os.path.join(os.path.abspath(__file__), 'shaders', shader_name)
        shader = Shader(shader_name, file_path)
        self._shader_cache[shader_name] = shader
        return shader

