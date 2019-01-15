import numpy as np

from .constants import *
from .shader_program import ShaderProgramCache
from .material import MetallicRoughnessMaterial
from .light import PointLight, SpotLight, DirectionalLight

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
        self._shadow_fb = None

        # Shader Program Cache
        self._program_cache = ShaderProgramCache()
        self._meshes = set()
        self._mesh_textures = set()
        self._shadow_textures = set()
        self._texture_alloc_idx = 0

        self._debug_quad_mesh = None

        # Set up framebuffer if needed / TODO

    def _render_debug_quad(self):
        if self._debug_quad_mesh is None:
            self._debug_quad_mesh = glGenVertexArrays(1)
            #verts = 0.99 * np.array([
            #    [-1.0, 1.0, 0.0, 0.0, 1.0],
            #    [-1.0, -1.0, 0.0, 0.0, 0.0],
            #    [1.0, 1.0, 0.0, 1.0, 1.0],
            #    [1.0, -1.0, 0.0, 1.0, 0.0],
            #])
            #vao = glGenVertexArrays(1)
            #vbo = glGenBuffers(1)
            #glBindVertexArray(vao)
            #glBindBuffer(GL_ARRAY_BUFFER, vbo)
            #glBufferData(GL_ARRAY_BUFFER, 4*verts.size, verts, GL_STATIC_DRAW)
            #glEnableVertexAttribArray(0);
            #glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(0))
            #glEnableVertexAttribArray(1)
            #glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(3*4))
            #self._debug_quad_mesh = (vao, vbo)
        glBindVertexArray(self._debug_quad_mesh)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

    def render(self, scene, flags):
        # If using shadow maps, render scene into each light's shadow map
        # first.

        # Update context with meshes and textures
        self._update_context(scene, flags)

        # Render necessary shadow maps
        if flags & RenderFlags.SHADOWS_DIRECTIONAL:
            for light_node in scene.directional_light_nodes:
                self._shadow_mapping_pass(scene, light_node, flags)
        if flags & RenderFlags.SHADOWS_SPOT:
            for light_node in scene.spot_light_nodes:
                self._shadow_mapping_pass(scene, light_node, flags)
        if flags & RenderFlags.SHADOWS_POINT:
            for light_node in scene.point_light_nodes:
                self._shadow_mapping_pass(scene, light_node, flags)

        # Make forward pass
        if False:
            # Render first light's texture
            # Get program
            for l in scene.directional_lights:
                if l.shadow_texture is not None:
                    self._configure_forward_pass_viewport(scene, 0)
                    program = self._get_debug_quad_program()
                    program._bind()
                    #program._print_uniforms()
                    self._bind_texture(l.shadow_texture, 'depthMap', program)

                    #for obj in scene.meshes:
                    #    for primitive in obj.primitives:
                    #        material = primitive.material
                    #        texture = material.baseColorTexture
                    #        if texture is not None:
                    #            print 'TEX'
                    #            self._bind_texture(texture, 'depthMap', program)

                    self._render_debug_quad()
                    self._reset_active_textures()
                    glFlush()
                    break
        else:
            return self._forward_pass(scene, flags)

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
                p.delete()

        self._meshes = scene_meshes.copy()

        # Update mesh textures
        mesh_textures = set()
        for m in scene_meshes:
            for p in m.primitives:
                mesh_textures |= p.material.textures

        # Add new textures to context
        for texture in mesh_textures - self._mesh_textures:
            texture._add_to_context()

        # Remove old textures from context
        for texture in self._mesh_textures - mesh_textures:
            texture.delete()

        self._mesh_textures = mesh_textures.copy()

        shadow_textures = set()
        for l in scene.lights:
            # Create if needed
            active = False
            if isinstance(l, DirectionalLight) and flags & RenderFlags.SHADOWS_DIRECTIONAL:
                active = True
            elif isinstance(l, PointLight) and flags & RenderFlags.SHADOWS_POINT:
                active = True
            if isinstance(l, SpotLight) and flags & RenderFlags.SHADOWS_SPOT:
                active = True

            if active and l.shadow_texture is None:
                    l.generate_shadow_texture()
            if l.shadow_texture is not None:
                shadow_textures.add(l.shadow_texture)

        # Add new textures to context
        for texture in shadow_textures - self._shadow_textures:
            texture._add_to_context()

        # Remove old textures from context
        for texture in self._shadow_textures - shadow_textures:
            texture.delete()

        self._shadow_textures = shadow_textures.copy()

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

    def _configure_shadow_mapping_viewport(self, light, flags):
        self._configure_shadow_framebuffer()
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fb)
        light.shadow_texture._bind()
        light.shadow_texture._bind_as_depth_attachment()
        glActiveTexture(GL_TEXTURE0)
        light.shadow_texture._bind()
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        glClear(GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, SHADOW_TEX_SZ, SHADOW_TEX_SZ)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)
        glDisable(GL_CULL_FACE)

    def delete(self):
        # Free shaders
        self._program_cache.clear()

        # Free meshes
        for mesh in self._meshes:
            for p in mesh.primitives:
                p.delete()

        # Free textures
        for mesh_texture in self._mesh_textures:
            mesh_texture.delete()

        for shadow_texture in self._shadow_textures:
            shadow_texture.delete()

        self._meshes = set()
        self._mesh_textures = set()
        self._shadow_textures = set()
        self._texture_alloc_idx = 0

        self._delete_main_framebuffer()
        self._delete_shadow_framebuffer()

    def _configure_shadow_framebuffer(self):
        if self._shadow_fb is None:
            self._shadow_fb = glGenFramebuffers(1)

    def _delete_shadow_framebuffer(self):
        if self._shadow_fb is not None:
            glDeleteFramebuffers(1, [self._shadow_fb])

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

    #def _get_camera_matrices(self, scene):
    #    main_camera_node = scene.main_camera_node
    #    if main_camera_node is None:
    #        raise ValueError('Cannot render scene without a camera')
    #    P = main_camera_node.camera.\
    #            get_projection_matrix(width=self.viewport_width, height=self.viewport_height)
    #    pose = scene.get_pose(main_camera_node)
    #    V = np.linalg.inv(pose) # V maps from world to camera
    #    print '---'
    #    print P
    #    print V
    #    return V, P

    def _get_camera_matrices(self, scene):
        if True:
            main_camera_node = scene.main_camera_node
            if main_camera_node is None:
                raise ValueError('Cannot render scene without a camera')
            P = main_camera_node.camera.\
                    get_projection_matrix(width=self.viewport_width, height=self.viewport_height)
            pose = scene.get_pose(main_camera_node)
            V = np.linalg.inv(pose) # V maps from world to camera
            print '---'
            print P
            print V
            return V, P
        else:
            #pose = scene.get_pose(light_node).copy()
            ## If a directional light, back the pose up from the scene centroid along
            ## the origin
            light_node = list(scene.directional_light_nodes)[0]
            light = light_node.light
            pose = scene.get_pose(light_node).copy()
            camera = light.get_shadow_camera(scene.scale)
            P = camera.get_projection_matrix()
            if isinstance(light, DirectionalLight):
                z_axis = pose[:3,2]
                loc = scene.centroid - z_axis * scene.scale
                pose[:3,3] = loc

            # Flip the z-axis so that the camera points at the scene
            pose[:3,0] = -pose[:3,0]
            pose[:3,2] = -pose[:3,2]
            V = np.linalg.inv(pose) # V maps from world to camera

            print '==='
            print P.dtype, V.dtype
            print P
            print V

            main_camera_node = scene.main_camera_node
            if main_camera_node is None:
                raise ValueError('Cannot render scene without a camera')
            #P = main_camera_node.camera.\
            #        get_projection_matrix(width=self.viewport_width, height=self.viewport_height)
            pose = scene.get_pose(main_camera_node)
            V = np.linalg.inv(pose) # V maps from world to camera
            print '---'
            print P.dtype, V.dtype
            print P
            print V
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
            program.set_uniform(b.format('emissive_factor'),
                                 material.emissiveFactor)
            if isinstance(material, MetallicRoughnessMaterial):
                program.set_uniform(b.format('base_color_factor'),
                                     material.baseColorFactor)
                program.set_uniform(b.format('metallic_factor'),
                                     material.metallicFactor)
                program.set_uniform(b.format('roughness_factor'),
                                     material.roughnessFactor)
            elif isinstance(material, SpecularGlossinessMaterial):
                program.set_uniform(b.format('diffuse_factor'),
                                     material.diffuseFactor)
                program.set_uniform(b.format('specular_factor'),
                                     material.specularFactor)
                program.set_uniform(b.format('glossiness_factor'),
                                     material.glossinessFactor)

            # Set blending options
            if material.alphaMode == 'BLEND':
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            else:
                glBlendFunc(GL_ONE, GL_ZERO)

            # Set wireframe mode
            wf = material.wireframe
            if flags & RenderFlags.FLIP_WIREFRAME:
                wf = not wf
            if (flags & RenderFlags.ALL_WIREFRAME) or wf:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Set culling mode
            if material.doubleSided:
                glDisable(GL_CULL_FACE)
            else:
                glEnable(GL_CULL_FACE)
                glCullFace(GL_BACK)
        else:
            glBlendFunc(GL_ONE, GL_ZERO)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Render mesh
        n_instances = 1
        if primitive.poses is not None:
            n_instances = len(primitive.poses)

        #program._print_uniforms()
        if primitive.indices is not None:
            #glDrawArraysInstanced(primitive.mode, 0, 3*len(primitive.positions), n_instances)
            glDrawElementsInstanced(primitive.mode, primitive.indices.size, GL_UNSIGNED_INT,
                                    ctypes.c_void_p(0), n_instances)
        else:
            glDrawArraysInstanced(primitive.mode, 0, len(primitive.positions), n_instances)

        # Unbind mesh buffers
        primitive._unbind()

    def _shadow_mapping_pass(self, scene, light_node, flags):
        light = light_node.light

        self._configure_shadow_mapping_viewport(light, flags)

        # Get program
        program = self._get_shadow_program(light, flags)
        program._bind()
        V, P = self._get_light_matrix(scene, light_node, flags)
        #program.set_uniform('light_matrix', self._get_light_matrix(scene, light_node, flags))

        # Draw objects
        for node in self._sorted_mesh_nodes(scene):
            mesh = node.mesh

            # Skip the mesh if it's not visible
            if not mesh.is_visible:
                continue

            for primitive in mesh.primitives:
                # First, get and bind the appropriate program
                program = self._get_primitive_program(primitive, RenderFlags.DEPTH_ONLY)
                program._bind()

                # Set the camera uniforms
                program.set_uniform('V', V)
                program.set_uniform('P', P)

                # Draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=RenderFlags.DEPTH_ONLY
                )

        program._unbind()
        glFlush()

    def _shadow_mapping_pass(self, scene, light_node, flags):

        light = light_node.light

        # Set up viewport for render
        self._configure_shadow_mapping_viewport(light, flags)
        #self._configure_forward_pass_viewport(scene, flags)

        # Set up camera matrices
        V, P = self._get_light_matrix(scene, light_node, flags)

        # Now, render each object in sorted order
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
                program.set_uniform('cam_pos', scene.get_pose(scene.main_camera_node)[:3,3])

                # Next, bind the lighting
                if not flags & RenderFlags.DEPTH_ONLY:
                    self._bind_lighting(scene, program, flags)

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=flags
                )
                self._reset_active_textures()

        # Unbind the shader and flush the output
        if program is not None:
            program._unbind()
        glFlush()

        # If doing offscreen render, copy result from framebuffer and return
        if flags & RenderFlags.OFFSCREEN:
            return self._read_main_framebuffer(scene)
        else:
            return None, None
    def _forward_pass(self, scene, flags):

        # Set up viewport for render
        self._configure_forward_pass_viewport(scene, flags)

        # Set up camera matrices
        V, P = self._get_camera_matrices(scene)

        # Now, render each object in sorted order
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
                program.set_uniform('cam_pos', scene.get_pose(scene.main_camera_node)[:3,3])

                # Next, bind the lighting
                if not flags & RenderFlags.DEPTH_ONLY:
                    self._bind_lighting(scene, program, flags)

                # Finally, bind and draw the primitive
                self._bind_and_draw_primitive(
                    primitive=primitive,
                    pose=scene.get_pose(node),
                    program=program,
                    flags=flags
                )
                self._reset_active_textures()

        # Unbind the shader and flush the output
        if program is not None:
            program._unbind()
        glFlush()

        # If doing offscreen render, copy result from framebuffer and return
        if flags & RenderFlags.OFFSCREEN:
            return self._read_main_framebuffer(scene)
        else:
            return None, None


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
            if l.range is not None:
                program.set_uniform(b + 'range', l.range)
            else:
                program.set_uniform(b + 'range', 0)
            program.set_uniform(b + 'position', position)
            if flags & RenderFlags.SHADOWS_POINT:
                raise NotImplementedError('Point light shadows not yet implemented')

        program.set_uniform('n_spot_lights', len(scene.spot_light_nodes))
        for i, n in enumerate(scene.spot_light_nodes):
            b = 'spot_lights[{}].'.format(i)
            l = n.light
            position = scene.get_pose(n)[:3,3]
            direction = scene.get_pose(n)[:3,2]
            program.set_uniform(b + 'color', l.color)
            program.set_uniform(b + 'intensity', l.intensity)
            if l.range is not None:
                program.set_uniform(b + 'range', l.range)
            else:
                program.set_uniform(b + 'range', 0)
            program.set_uniform(b + 'position', position)
            program.set_uniform(b + 'direction', direction)
            program.set_uniform(b + 'inner_cone_angle', l.innerConeAngle)
            program.set_uniform(b + 'outer_cone_angle', l.outerConeAngle)
            if flags & RenderFlags.SHADOWS_SPOT:
                self._bind_texture(l.shadow_texture, b + 'shadow_map', program)
                V, P = self._get_light_matrix(scene, n, flags)
                lm = P.dot(V)
                program.set_uniform(b + 'light_matrix', lm)

        program.set_uniform('n_directional_lights', len(scene.directional_light_nodes))
        for i, n in enumerate(scene.directional_light_nodes):
            b = 'directional_lights[{}].'.format(i)
            l = n.light
            direction = scene.get_pose(n)[:3,2]
            program.set_uniform(b + 'color', l.color)
            program.set_uniform(b + 'intensity', l.intensity)
            program.set_uniform(b + 'direction', direction)
            if flags & RenderFlags.SHADOWS_DIRECTIONAL:
                self._bind_texture(l.shadow_texture, b + 'shadow_map', program)
                V, P = self._get_light_matrix(scene, n, flags)
                lm = P.dot(V)
                program.set_uniform(b + 'light_matrix', lm)

    def _get_light_matrix(self, scene, light_node, flags):
        light_node = list(scene.directional_light_nodes)[0]
        light = light_node.light
        pose = scene.get_pose(light_node).copy()
        camera = light.get_shadow_camera(scene.scale)
        P = camera.get_projection_matrix()
        if isinstance(light, DirectionalLight):
            z_axis = pose[:3,2]
            loc = scene.centroid - z_axis * scene.scale
            pose[:3,3] = loc

        # Flip the z-axis so that the camera points at the scene
        pose[:3,0] = -pose[:3,0]
        pose[:3,2] = -pose[:3,2]
        V = np.linalg.inv(pose) # V maps from world to camera
        #return P.dot(V)
        return V, P

    def _get_shadow_program(self, light, flags):
        if isinstance(light, DirectionalLight) or isinstance(light, SpotLight):
            program = self._program_cache.get_program(
                vertex_shader='shadow_depth_vert.glsl',
                fragment_shader='shadow_depth_frag.glsl',
            )
            if not program._in_context():
                program._add_to_context()
            return program
        else:
            raise NotImplementedError('Point light shadows not yet implemented')

    def _get_debug_quad_program(self):
        program = self._program_cache.get_program(
            vertex_shader='debug_quad_vert.glsl',
            fragment_shader='debug_quad_frag.glsl'
        )
        if not program._in_context():
            program._add_to_context()
        return program

    def _get_primitive_program(self, primitive, flags):
        vertex_shader = None
        fragment_shader = None
        geometry_shader = None
        defines = {}
        if flags & RenderFlags.DEPTH_ONLY:
            vertex_shader = 'mesh_depth_vert.glsl'
            fragment_shader = 'mesh_depth_frag.glsl'
            bf = primitive.buf_flags
            buf_idx = 1
            if bf & BufFlags.NORMAL:
                defines['NORMAL_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.TANGENT:
                defines['TANGENT_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.TEXCOORD_0:
                defines['TEXCOORD_0_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.TEXCOORD_1:
                defines['TEXCOORD_1_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.COLOR_0:
                defines['COLOR_0_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.JOINTS_0:
                defines['JOINTS_0_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.WEIGHTS_0:
                defines['WEIGHTS_0_LOC'] = buf_idx
                buf_idx += 1
            defines['INST_M_LOC'] = buf_idx
        else:
            vertex_shader = 'mesh_vert.glsl'
            fragment_shader = 'mesh_frag.glsl'
            tf = primitive.material.tex_flags
            bf = primitive.buf_flags
            if tf & TexFlags.NORMAL:
                defines['HAS_NORMAL_TEX'] = 1
            if tf & TexFlags.OCCLUSION:
                defines['HAS_OCCLUSION_TEX'] = 1
            if tf & TexFlags.EMISSIVE:
                defines['HAS_EMISSIVE_TEX'] = 1
            if tf & TexFlags.BASE_COLOR:
                defines['HAS_BASE_COLOR_TEX'] = 1
            if tf & TexFlags.METALLIC_ROUGHNESS:
                defines['HAS_METALLIC_ROUGHNESS_TEX'] = 1
            if tf & TexFlags.DIFFUSE:
                defines['HAS_DIFFUSE_TEX'] = 1
            if tf & TexFlags.SPECULAR_GLOSSINESS:
                defines['HAS_SPECULAR_GLOSSINESS_TEX'] = 1
            if isinstance(primitive.material, MetallicRoughnessMaterial):
                defines['USE_METALLIC_MATERIAL'] = 1
            elif isinstance(material, SpecularGlossinessMaterial):
                defines['USE_GLOSSY_MATERIAL'] = 1
            buf_idx = 1
            if bf & BufFlags.NORMAL:
                defines['NORMAL_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.TANGENT:
                defines['TANGENT_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.TEXCOORD_0:
                defines['TEXCOORD_0_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.TEXCOORD_1:
                defines['TEXCOORD_1_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.COLOR_0:
                defines['COLOR_0_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.JOINTS_0:
                defines['JOINTS_0_LOC'] = buf_idx
                buf_idx += 1
            if bf & BufFlags.WEIGHTS_0:
                defines['WEIGHTS_0_LOC'] = buf_idx
                buf_idx += 1
            defines['INST_M_LOC'] = buf_idx
            if flags & RenderFlags.SHADOWS_SPOT:
                defines['SPOT_LIGHT_SHADOWS'] = 1
            if flags & RenderFlags.SHADOWS_DIRECTIONAL:
                defines['DIRECTIONAL_LIGHT_SHADOWS'] = 1
            if flags & RenderFlags.SHADOWS_POINT:
                defines['POINT_LIGHT_SHADOWS'] = 1
        program = self._program_cache.get_program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            defines=defines
        )
        if not program._in_context():
            program._add_to_context()
        return program
