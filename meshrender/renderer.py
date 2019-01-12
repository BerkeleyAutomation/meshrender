import numpy as np

from .constants import *
from .shader_program import ShaderProgramCache
from .material import MetallicRoughnessMaterial

from OpenGL.GL import *

class Renderer(object):
    """Class for handling all rendering operations on a scene.

    Note
    ----
    This doesn't handle creating an OpenGL context -- it assumes that
    the context has already been created.
    """

    def __init__(self, viewport_width, viewport_height):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        # Optional framebuffer for offscreen renders
        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._main_fb_dims = (None, None)

        # Shader Program Cache
        self._program_cache = ShaderProgramCache()
        self._meshes = set()
        self._textures = set()
        self._texture_alloc_idx = 0

        # Set up framebuffer if needed / TODO

    def render(self, scene, flags):
        # If using shadow maps, render scene into each light's shadow map
        # first.
        if flags & RenderFlags.SHADOWS:
            raise NotImplementedError('Shadows not yet implemented')

        # Else, set up normal render
        else:
            self._forward_pass(scene, flags)
            self._reset_active_textures()


    def _update_context(self, scene, flags):

        # Update meshes
        scene_meshes = scene.meshes

        # Add new meshes to context
        for mesh in scene_meshes - self._meshes:
            for p in mesh.primitives:
                p._add_to_context()

        # Remove old meshes from context
        for mesh in self._meshes - scene_meshes:
            for p in mesh.primitives:
                p._unbind()
                p._remove_from_context()

        self._meshes = scene_meshes.copy()

        # Update textures
        scene_textures = set()
        for m in scene_meshes:
            for p in m.primitives:
                scene_textures |= p.material.textures
        for l in scene.lights:
            if l.castsShadows:
                scene_textures.add(l.depth_texture)

        # Add new textures to context
        for texture in scene_textures - self._textures:
            texture._add_to_context()

        for texture in self._textures - scene_textures:
            texture._unbind()
            texture._remove_from_context()

        # Update shaders
        # TODO

    def _configure_forward_pass_viewport(self, scene, flags):

        # If using offscreen render, bind main framebuffer
        if flags & RenderFlags.OFFSCREEN:
            self._configure_main_framebuffer()
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
        else:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

        glClearColor(*scene.bg_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, self.viewport_width, self.viewport_height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

    def delete(self):
        # Free shaders
        self._program_cache.clear()

        # Free meshes
        for mesh in self._meshes:
            mesh.delete()

        # Free textures
        for texture in self._textures:
            texture.delete()

        self._meshes = set()
        self._textures = set()

        self._delete_main_framebuffer()

    def _configure_main_framebuffer(self):
        # If mismatch with prior framebuffer, delete it
        if (self._main_fb is not None and
            self.viewport_width != self._main_fb_dims[0] or
            self.viewport_height != self._main_fb_dims[1]):
            self._delete_main_framebuffer()

        # If framebuffer doesn't exist, create it
        if self._main_fb is None:
            self._main_cb, self._main_db = glGenRenderbuffers(2)
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_cb)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, self.width, self.height)
            glBindRenderbuffer(GL_RENDERBUFFER, self._main_db)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
            self._main_fb = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._main_cb)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._main_db)
            self._main_fb_dims = (self.viewport_width, self.viewport_height)

    def _delete_main_framebuffer(self):
        if self._main_fb is not None:
            glDeleteFramebuffers(1, [self._main_fb])
        if self._main_cb is not None:
            glDeleteRenderbuffers(1, [self._main_cb])
        if self._main_db is not None:
            glDeleteRenderbuffers(1, [self._main_db])

        self._main_fb = None
        self._main_cb = None
        self._main_db = None
        self._main_fb_dims = (None, None)

    def _get_camera_matrices(self, scene):
        main_camera_node = scene.main_camera_node
        if main_camera_node is None:
            raise ValueError('Cannot render scene without a camera')
        P = main_camera_node.camera.\
                get_projection_matrix(width=self.viewport_width, height=self.viewport_height)
        pose = scene.get_pose(main_camera_node)
        V = np.linalg.inv(pose) # V maps from world to camera
        return V, P

    def _bind_texture(self, texture, uniform_name, program):
        """Bind a texture to an active texture unit and return
        the texture unit index that was used.
        """
        tex_id = self._get_next_active_texture()
        glActiveTexture(GL_TEXTURE0 + tex_id)
        texture._bind()
        program.set_uniform('material.{}'.format(uniform_name), tex_id)

    def _get_next_active_texture(self):
        val = self._texture_alloc_idx
        self._texture_alloc_idx += 1
        return val

    def _reset_active_textures(self):
        self._texture_alloc_idx = 0

    def _bind_and_draw_primitive(self, primitive, pose, program, flags):
        # Set model pose matrix
        program.set_uniform('M', pose)

        # Bind mesh buffers
        primitive._bind()

        # Bind mesh material
        if not flags & RenderFlags.DEPTH_ONLY:
            material = primitive.material

            # Bind textures
            tf = material.tex_flags
            if tf & TexFlags.NORMAL:
                self._bind_texture(material.normalTexture,
                                   'normal_texture', program)
            if tf & TexFlags.OCCLUSION:
                self._bind_texture(material.occlusionTexture,
                                   'occlusion_texture', program)
            if tf & TexFlags.EMISSIVE:
                self._bind_texture(material.emissiveTexture,
                                   'emissive_texture', program)
            if tf & TexFlags.BASE_COLOR:
                self._bind_texture(material.baseColorTexture,
                                   'base_color_texture', program)
            if tf & TexFlags.METALLIC_ROUGHNESS:
                self._bind_texture(material.metallicRoughnessTexture,
                                   'metallic_roughness_texture', program)
            if tf & TexFlags.DIFFUSE:
                self._bind_texture(material.diffuseTexture,
                                   'diffuse_texture', program)
            if tf & TexFlags.SPECULAR_GLOSSINESS:
                self._bind_texture(material.specularGlossinessTexture,
                                   'specular_glossiness_texture', program)

            # Bind other uniforms
            b = 'material.{}'
            program.bind_uniform(b.format('emissive_factor'),
                                 material.emissiveFactor)
            if isinstance(material, MetallicRoughnessMaterial):
                program.bind_uniform(b.format('base_color_factor'),
                                     material.baseColorFactor)
                program.bind_uniform(b.format('metallic_factor'),
                                     material.metallicFactor)
                program.bind_uniform(b.format('roughness_factor'),
                                     material.roughnessFactor)
            elif isinstance(material, SpecularGlossinessMaterial):
                program.bind_uniform(b.format('diffuse_factor'),
                                     material.diffuseFactor)
                program.bind_uniform(b.format('specular_factor'),
                                     material.specularFactor)
                program.bind_uniform(b.format('glossiness_factor'),
                                     material.glossinessFactor)

            # Set blending options
            if material.alphaMode == 'BLEND':
                glBlendFunction(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            else:
                glBlendFunction(GL_ONE, GL_ZERO)

            # Set wireframe mode
            if material.wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glBlendFunction(GL_ONE, GL_ZERO)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Render mesh
        n_instances = 1
        if primitive.poses is not None:
            n_instances = len(primitive.poses)

        if primitive.indices is not None:
            glDrawElementsInstanced(primitive.mode, obj.indices.size, GL_UNSIGNED_INT,
                                    ctypes.c_void_p(0), n_instances)
        else:
            glDrawArraysInstanced(primitive.mode, 0, len(obj.positions), n_instances)

        # Unbind mesh buffers
        primitive._unbind()

    def _forward_pass(self, scene, flags):
        # Update context with meshes and textures
        self._update_context(scene, flags)

        # Set up viewport for render
        self._configure_forward_pass_viewport(scene, flags)

        # Set up camera matrices
        V, P = self._get_camera_matrices(scene)

        # Now, render each object in sorted order
        prior_program = None
        for node in self._sorted_mesh_nodes(scene):
            mesh = node.mesh

            # Skip the mesh if it's not visible
            if not mesh.is_visible:
                continue

            for primitive in mesh.primitives:

                # First, get and bind the appropriate program
                program = self._get_primitive_program(primitive, flags) # TODO
                program._bind()

                # Set the camera uniforms
                program.set_uniform('V', V)
                program.set_uniform('P', P)

                # Next, bind the lighting
                if program != prior_program:
                    if not flags & RenderFlags.DEPTH_ONLY:
                        self._bind_lighting(scene, program, flags)

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=flags
                )

                prior_program = program

        # Unbind the shader and flush the output
        if prior_program is not None:
            prior_program._unbind()
        glFlush()

        # If doing offscreen render, copy result from framebuffer and return

    def _read_main_framebuffer(self, scene):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)
        color_buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        depth_buf = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)

        color_im = np.frombuffer(color_buf, dtype=np.uint8).reshape((height, width, 3))
        color_im = np.flip(color_im, axis=0)

        depth_im = np.frombuffer(depth_buf, dtype=np.float32).reshape((height, width))
        depth_im = np.flip(depth_im, axis=0)

        inf_inds = (depth_im == 1.0)
        depth_im = 2.0 * depth_im - 1.0
        z_near, z_far = scene.main_camera_node.camera.znear, scene.main_camera_node.camera.zfar
        depth_im = 2.0 * z_near * z_far / (z_far + z_near - depth_im * (z_far - z_near))
        depth_im[inf_inds] = 0.0

        return color_im, depth_im

    def _sorted_mesh_nodes(self, scene):
        cam_loc = scene.get_pose(scene.main_camera_node)[:3,3]
        solid_nodes = []
        trans_nodes = []
        for node in scene.mesh_nodes:
            mesh = node.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        # TODO BETTER SORTING METHOD
        trans_nodes.sort(
            key=lambda n : -np.linalg.norm(scene.get_pose(n)[:3,3] - cam_loc)
        )
        solid_nodes.sort(
            key=lambda n : -np.linalg.norm(scene.get_pose(n)[:3,3] - cam_loc)
        )

        return solid_nodes + trans_nodes

    def _bind_lighting(self, scene, program, flags):
        """Bind all lighting uniform values for a scene.
        """
        # TODO handle shadow map textures

        program.set_uniform('n_point_lights', len(scene.point_light_nodes))
        for i, n in enumerate(scene.point_light_nodes):
            b = 'point_lights[{}].'.format(i)
            l = n.light
            position = scene.get_pose(n)[:3,3]
            program.set_uniform(b + 'color', l.color)
            program.set_uniform(b + 'intensity', l.intensity)
            program.set_uniform(b + 'range', l.range)
            program.set_uniform(b + 'position', position)

        program.set_uniform('n_spot_lights', len(scene.spot_light_nodes))
        for i, n in enumerate(scene.spot_light_nodes):
            b = 'spot_lights[{}].'.format(i)
            l = n.light
            position = scene.get_pose(n)[:3,3]
            direction = scene.get_pose(n)[:3,2]
            program.set_uniform(b + 'color', l.color)
            program.set_uniform(b + 'intensity', l.intensity)
            program.set_uniform(b + 'range', l.range)
            program.set_uniform(b + 'position', position)
            program.set_uniform(b + 'direction', direction)
            program.set_uniform(b + 'inner_cone_angle', l.innerConeAngle)
            program.set_uniform(b + 'outer_cone_angle', l.outerConeAngle)

        program.set_uniform('n_directional_lights', len(scene.directional_light_nodes))
        for i, n in enumerate(scene.directional_light_nodes):
            b = 'directional_lights[{}].'.format(i)
            l = n.light
            direction = scene.get_pose(n)[:3,2]
            program.set_uniform(b + 'color', l.color)
            program.set_uniform(b + 'intensity', l.intensity)
            program.set_uniform(b + 'range', l.range)
            program.set_uniform(b + 'direction', direction)

    def _get_primitive_program(self, primitive, flags):
        shader_filenames = []
        defines = []
        if flags & RenderFlags.DEPTH_ONLY:
            pass
        else:
            shader_filenames.extend(['mesh_vert.glsl', 'mesh_frag.glsl'])

            tf = primitive.material.tex_flags
            bf = primitive.buf_flags
            if tf & TexFlags.NORMAL:
                defines.append('NORMAL')
            if tf & TexFlags.OCCLUSION:
                defines.append('OCCLUSION')
            if tf & TexFlags.EMISSIVE:
                defines.append('EMISSIVE')
            if tf & TexFlags.BASE_COLOR:
                defines.append('BASE_COLOR')
            if tf & TexFlags.METALLIC_ROUGHNESS:
                defines.append('METALLIC_ROUGHNESS')
            if tf & TexFlags.DIFFUSE:
                defines.append('DIFFUSE')
            if tf & TexFlags.SPECULAR_GLOSSINESS:
                defines.append('SPECULAR_GLOSSINESS')
            if isinstance(primitive.material, MetallicRoughnessMaterial):
                defines.append('METALLIC_MATERIAL')
            elif isinstance(material, SpecularGlossinessMaterial):
                defines.append('GLOSSINESS_MATERIAL')
            if bf & BufFlags.NORMAL:
                defines.append('HAS_NORMALS')
            if bf & BufFlags.TANGENT:
                defines.append('HAS_TANGENTS')
            if bf & BufFlags.TEXCOORD_0:
                defines.append('HAS_TEXCOORD_0')
            if bf & BufFlags.TEXCOORD_1:
                defines.append('HAS_TEXCOORD_1')
            if bf & BufFlags.COLOR_0:
                defines.append('HAS_COLOR_0')
            if bf & BufFlags.JOINTS_0:
                defines.append('HAS_JOINTS_0')
            if bf & BufFlags.WEIGHTS_0:
                defines.append('HAS_WEIGHTS_0')
        import pdb; pdb.set_trace()
        program = self._program_cache.get_program(shader_filenames, defines)
        if not program._in_context:
            program._add_to_context()
        return program
