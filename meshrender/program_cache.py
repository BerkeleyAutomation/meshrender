class ProgramCache(object):

    def __init__(self):
        self._program_cache = {}

    def get_program(self, shader_filenames):
        shader_names = []
        for fn in shader_filenames:
            _, name = os.path.split(fn)
            shader_names.append(name)
        shader_names = tuple(sorted(shader_names))

        if shader_names not in self._program_cache:
            self._program_cache[shader_names] = Program(shader_filenames)
        return self._program_cache[shader_names]

    def delete(self):
        for key in self._program_cache:
            self._program_cache[key].delete()
        self._program_cache = {}
