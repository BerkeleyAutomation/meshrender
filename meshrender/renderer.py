import numpy as np

from .constants import *
from .shader_program import ShaderProgramCache

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

        # Set up framebuffer if needed / TODO

    def render(self, scene, render_flags):
        self._texture_count = 0
        self._pre_obj_texture_count = 0

        # Delete old textures
        self._delete_old_textures(scene)

        # If using shadow maps, render scene into each light's shadow map
        # first.
        if render_flags & RenderFlags.SHADOWS:
            raise NotImplementedError('Shadows not yet implemented')

        # Else, set up normal render
        else:
            self._standard_forward_pass(scene, render_flags)


    def _update_context(self, scene, flags):

        # Update meshes
        scene_meshes = scene.meshes

        # Add new meshes to context
        for mesh in scene_meshes - self._meshes:
            mesh._add_to_context()

        # Remove old meshes from context
        for mesh in self._meshes - scene_meshes:
            mesh._unbind()
            mesh._remove_from_context()

        self._meshes = scene_meshes.copy()

        # Update textures
        scene_textures = set()
        for m in scene_meshes:
            scene_textures |= mesh.textures
        for l in scene.lights:
            if l.casts_shadows:
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
        P = node.camera.get_projection_matrix(width=self.viewport_width, height=self.viewport_height)
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

    def _draw_primitive(self, primitive, pose, program, flags):
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





        # Render mesh


    def _forward_pass(self, scene, flags):
        # Update context with meshes and textures
        self._update_context(scene, flags)

        # Set up viewport for render
        self._configure_forward_pass_viewport()

        # Set up camera matrices
        V, P = self._get_camera_matrices(scene)

        # Now, render each object in sorted order
        prior_program = None
        for node in self._sorted_mesh_nodes(scene):
            mesh = node.mesh
            # Skip the mesh if it's not visible
            if not mesh.is_visible:
                continue


            # First, get the appropriate program
            program = self._get_program(obj) # TODO
            program.bind()

            # Next, bind the object and its materials
            obj._bind()
            # Bind material properties unless using a depth-only render.
            if not (render_flags & RenderFlags.DEPTH_ONLY):
                self._bind_object_material(name, obj, program)

            # Set MVP matrices
            M = scene.get_pose(node)
            program.set_uniform('M', M)
            program.set_uniform('V', V)
            program.set_uniform('P', P)

            # Next, set lighting if not doing a depth-only render and not using
            # prior program, which already had its uniforms bound for lighting.
            if program != prior_program:
                if not render_flags & RenderFlags.DEPTH_ONLY:
                    self._bind_lighting_uniforms(scene, program, render_flags)

            # Next, set the polygon mode if necessary
            if obj.material.wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            vaf = obj.vertex_array_flags

            n_instances = 1
            if vaf & VertexArrayFlags.INSTANCED:
                n_instances = len(obj.poses)
            if vaf & VertexArrayFlags.TRIANGLES:
                if vaf & VertexArrayFlags.ELEMENTS:
                    glDrawElementsInstanced(GL_TRIANGLES, 3*len(obj.faces), GL_UNSIGNED_INT,
                            ctypes.c_void_p(0), n_instances)
                else:
                    glDrawArraysInstanced(GL_TRIANGLES, 0, 3*len(obj.faces), n_instances)
            else:
                glDrawArraysInstanced(GL_POINTS, 0, len(obj.vertices), n_instances)

            prior_program = program
            #program._print_uniforms()

            # Unbind the object
            obj._unbind()

        # Unbind the shader and flush the output
        glUseProgram(0)
        glFlush()

        if not render_flags & RenderFlags.OFFSCREEN:
            return
        else:
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
            z_near, z_far = scene.camera.z_near, scene.camera.z_far
            depth_im = 2.0 * z_near * z_far / (z_far + z_near - depth_im * (z_far - z_near))
            depth_im[inf_inds] = 0.0

            return color_im, depth_im

    def _sorted_mesh_nodes(self, scene):
        cam_loc = scene.get_pose(scene.main_camera_node)[:3,3]
        solid_nodes = []
        trans_nodes = []
        for node in scene.mesh_nodes:
            mesh = scene.mesh
            if mesh.is_transparent:
                trans_nodes.append(node)
            else:
                solid_nodes.append(node)

        # TODO BETTER SORTING METHOD
        trans_nodes.sort(
            key=lambda n : -np.linalg.norm(scene.get_pose(n)[:3,3] - cam_loc)
        )
        solid_names.sort(
            key=lambda n : -np.linalg.norm(scene.get_pose(n)[:3,3] - cam_loc)
        )

        return solid_nodes + trans_nodes

    def _bind_object_material(self, name, obj, program):
        """Bind object's material to shader program uniforms.

        Parameters
        ----------
        name : str
            Name of object in scene.
        obj : :obj:`SceneObject`
            Scene object to bind materials for.
        program : :obj:`ShaderProgram`
            Shader program to bind materials for.
        """
        tm = obj.texture_flags

        if tm & TextureFlags.DIFFUSE:
            tex = self._texture_cache.get_texture(name='{}.diffuse'.format(name),
                                                  data=obj.material.diffuse)
            tex_idx = self._bind_texture(tex)
            program.set_uniform('material.diffuse', tex_idx)
        else:
            diffuse = obj.material.diffuse
            if len(diffuse) == 3:
                diffuse = np.hstack((diffuse, 1.0))
            program.set_uniform('material.diffuse', diffuse)

        if tm & TextureFlags.SPECULAR:
            tex = self._texture_cache.get_texture(name='{}.specular'.format(name),
                                                  data=obj.material.specular)
            tex_idx = self._bind_texture(tex)
            program.set_uniform('material.specular', tex_idx)
        else:
            program.set_uniform('material.specular', obj.material.specular)

        program.set_uniform('material.shininess', obj.material.shininess)

        if tm & TextureFlags.EMISSION:
            tex = self._texture_cache.get_texture(name='{}.emission'.format(name),
                                                  data=obj.material.emission)
            tex_idx = self._bind_texture(tex)
            program.set_uniform('material.emission', tex_idx)
        else:
            emission = obj.material.emission
            if emission is None:
                emission = np.zeros(3)
            program.set_uniform('material.emission', emission)

        if tm & TextureFlags.NORMAL:
            tex = self._texture_cache.get_texture(name='{}.normal'.format(name),
                                                  data=obj.material.normal)
            tex_idx = self._bind_texture(tex)
            program.set_uniform('material.normal', tex_idx)

        if tm & TextureFlags.HEIGHT:
            tex = self._texture_cache.get_texture(name='{}.height'.format(name),
                                                  data=obj.material.height)
            tex_idx = self._bind_texture(tex)
            program.set_uniform('material.height', tex_idx)


    def _delete_old_textures(self, scene):
        """Remove textures that aren't in use in the current scene.
        """
        names = set(scene.objects.keys()) | set(scene.lights.keys())
        for tn in self._texture_cache.texture_names:
            base = tn.split('.', 1)[0]
            if base not in names:
                self._texture_cache.delete_texture(tn)

    def _bind_texture(self, texture):
        """Bind a texture to an active texture unit and return
        the texture unit index that was used.
        """
        glActiveTexture(GL_TEXTURE0 + self._texture_count)
        texture.bind()
        old_tex_count = self._texture_count
        self._texture_count += 1
        return old_tex_count

    def _bind_lighting_uniforms(self, scene, program, render_flags):
        """Bind all lighting uniform values for a scene.
        """
        # TODO handle shadow map textures
        point_lights = scene.point_lights
        direc_lights = scene.directional_lights
        spot_lights = scene.spot_lights

        program.set_uniform('n_point_lights', len(scene.point_lights))
        for i, ln in enumerate(scene.point_lights.keys()):
            base = 'point_lights[{}].'.format(i)
            l = scene.point_lights[ln]
            pose = scene.get_pose(ln)
            position = pose[:3,:3].dot(l.position) + pose[:3,3]
            program.set_uniform(base + 'ambient', l.ambient)
            program.set_uniform(base + 'diffuse', l.diffuse)
            program.set_uniform(base + 'specular', l.specular)
            program.set_uniform(base + 'position', position)
            program.set_uniform(base + 'constant', l.constant)
            program.set_uniform(base + 'linear', l.linear)
            program.set_uniform(base + 'quadratic', l.quadratic)

        program.set_uniform('n_spot_lights', len(scene.spot_lights))
        for i, ln in enumerate(scene.spot_lights.keys()):
            base = 'spot_lights[{}].'.format(i)
            l = scene.spot_lights[ln]
            pose = scene.get_pose(ln)
            position = pose[:3,:3].dot(l.position) + pose[:3,3]
            direction = pose[:3,:3].dot(l.direction)
            program.set_uniform(base + 'ambient', l.ambient)
            program.set_uniform(base + 'diffuse', l.diffuse)
            program.set_uniform(base + 'specular', l.specular)
            program.set_uniform(base + 'position', position)
            program.set_uniform(base + 'direction', direction)
            program.set_uniform(base + 'constant', l.constant)
            program.set_uniform(base + 'linear', l.linear)
            program.set_uniform(base + 'quadratic', l.quadratic)
            program.set_uniform(base + 'inner_angle', np.deg2rad(l.inner_angle))
            program.set_uniform(base + 'outer_angle', np.deg2rad(l.outer_angle))

        program.set_uniform('n_direc_lights', len(scene.directional_lights))
        for i, ln in enumerate(scene.directional_lights.keys()):
            base = 'directional_lights[{}].'.format(i)
            l = scene.directional_lights[ln]
            pose = scene.get_pose(ln)
            direction = pose[:3,:3].dot(l.direction)
            program.set_uniform(base + 'ambient', l.ambient)
            program.set_uniform(base + 'diffuse', l.diffuse)
            program.set_uniform(base + 'specular', l.specular)
            program.set_uniform(base + 'direction', direction)

    def _init_standard_render(self, scene, render_flags):
        glClearColor(*scene.bg_color)
        glViewport(0, 0, scene.camera.intrinsics.width, scene.camera.intrinsics.height)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

    def _get_program(self, obj):
        vbf = obj.vertex_buffer_flags
        vaf = obj.vertex_array_flags
        tf = obj.texture_flags

        # Trimesh
        if vaf & VertexArrayFlags.TRIANGLES:
            # Per-Vertex Color only
            if vbf & VertexBufferFlags.COLOR:
                if not vbf & VertexBufferFlags.TEXTURE:
                    return self._program_cache.get_program(['vc_mesh.vert', 'bp_mesh.frag'])
                else:
                    raise NotImplementedError('STILL NEED TO IMP TEXTURE SHADERS')
            else:
                if vbf & VertexBufferFlags.TEXTURE:
                    # Singleton textures
                    if tf == TextureFlags.DIFFUSE:
                        return self._program_cache.get_program(['tex_mesh.vert', 'bp_d_tex_mesh.frag'])
                    else:
                        raise NotImplementedError('STILL NEED TO IMP TEXTURE SHADERS')
                else:
                    return self._program_cache.get_program(['simple_mesh.vert', 'bp_mesh.frag'])
        # Point Cloud
        else:
            return self._program_cache.get_program(['point_cloud.vert', 'point_cloud.frag'])

