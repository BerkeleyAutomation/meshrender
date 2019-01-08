
class Renderer(object):
    """Class for handling all rendering operations on a scene.

    Note
    ----
    This doesn't handle creating an OpenGL context -- it assumes that
    the context has already been created.
    """

    def __init__(self):
        # Optional framebuffer for offscreen renders
        self._framebuf = None
        self._colorbuf = None
        self._depthbuf = None
        self._framebuf_dims = (None, None)

        # Shader Program Cache
        self._program_cache = ShaderProgramCache()

        # Texture Cache
        self._texture_cache = TextureCache()

    def render(self, scene, render_flags, render_modes):
        self._texture_count = 0
        self._pre_obj_texture_count = 0

        # If using shadow maps, render scene into each light's shadow map
        # first.

        # Set up normal render

    def _standard_forward_pass(self, scene, render_flags):
        if render_flags & RenderFlags.OFFSCREEN:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._framebuf)
        else:
            glBindFramebuffer(0)

        # Clear viewport
        glViewport(0, 0, scene.camera.intrinsics.width, scene.camera.intrinsics.height)
        glClearColor(*scene.bg_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up camera matrices
        cam_pose = scene.get_pose(scene.camera_name)
        V = scene.camera.V(pose)
        P = scene.camera.P

        # Now, render each object
        prior_program = None
        for name in self._sorted_object_names(scene):
            obj = scene.objects[name]
            # Skip the object if it's not visible
            if not obj.is_visible:
                continue
            # SET GL BLEND MODE
            # First, get the appropriate program
            program = self._get_program(obj) # TODO
            # Next, set lighting if not doing a depth-only render and not using
            # prior program, which already had its uniforms bound for lighting.
            if program != prior_program:
                if not render_flags & RenderFlags.DEPTH_ONLY:
                    self._bind_lighting_uniforms(scene, program, render_flags)


            # same shader
            # Next, bind the object and its materials
            self._bind_object(name, scene, program, render_flags)

            # Next, set the polygon mode if necessary
            if obj.material.wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            vaf = obj.vertex_array_flags

            if vaf & VertexArrayFlags.TRIANGLES:
                if vaf & VertexArrayFlags.INSTANCED:
                    if vaf & VertexArrayFlags.ELEMENTS:
                        glDrawElementsInstanced(GL_TRIANGLES, 3*len(obj.faces), GL_UNSIGNED_INT,
                                ctypes.c_void_p(0), len(obj.poses))
                    else:
                        glDrawArraysInstanced(GL_TRIANGLES, 0, 3*len(obj.faces), len(obj.poses))
                else:
                    if vaf & VertexArrayFlags.ELEMENTS:
                    else:
                        glDrawArrays(GL_TRIANGLES, 0, 3*len(obj.faces))
            else:
                if vaf & VertexArrayFlags.INSTANCED:
                    glDrawArraysInstanced(GL_POINTS, 0, len(obj.vertices), len(obj.poses))
                else:
                    glDrawArrays(GL_POINTS, 0, len(obj.vertices))

            prior_program = program

    def _sorted_object_names(self, scene):
        """Sort object names so that we iterate front-to-back on solid objects,
        and then back-to-front on transparent ones.
        """
        cam_loc = scene.get_pose(scene.camera_name)[:3,3]

        solid_names = []
        transparent_names = []
        for obj_name in scene.objects:
            obj = scene.objects[obj_name]
            if obj.is_transparent:
                transparent_names.append(obj_name)
            else:
                solid_names.append(obj_name)

        transparent_names.sort(
            key=lambda n : -np.linalg.norm(scene.objects[n].bbox_center - cam_loc)
        )
        solid_names.sort(
            key=lambda n : -np.linalg.norm(scene.objects[n].bbox_center - cam_loc)
        )

        return solid_names + transparent_names

    def _bind_camera(self, name='camera', camera, scene, program, render_flags):
        pass

    def _bind_object(self, name, scene, program, render_flags):
        """Bind a scene object to a shader program's uniforms.

        Parameters
        ----------
        name : str
            Name of object in scene.
        scene : :obj:`Scene`
            Scene containing object.
        program : :obj:`ShaderProgram`
            Shader program to bind materials for.
        render_flags : int
            Flags for rendering
        """
        obj = scene.objects[name]
        obj.bind()

        # Bind material properties unless using a depth-only render.
        if render_flags & RenderFlags.DEPTH_ONLY:
            self._bind_object_material(name, obj, program)

        # Bind M matrix
        M = scene.get_pose(name)
        program.set_uniform('M', M)

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
            program.set_uniform('material.diffuse', obj.material.diffuse)

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

    def _init_framebuffer(self, scene):
        camera = scene.camera

        # If mismatch with prior framebuffer, delete it
        if (self._framebuf is not None and
            camera.intrinsics.width != self._framebuf_dims[0] or
            camera.intrinsics.height != self._framebuf_dims[1]):
            self._delete_framebuffer()

        # If framebuffer doesn't exist, create it
        if self._framebuf is None:
            self._colorbuf, self._depthbuf = glGenRenderbuffers(2)
            glBindRenderbuffer(GL_RENDERBUFFER, self._colorbuf)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, self.width, self.height)
            glBindRenderbuffer(GL_RENDERBUFFER, self._depthbuf)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
            self._framebuf = glGenFramebuffers(1)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._framebuf)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._colorbuf)
            glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depthbuf)
            self._framebuf_dims = (camera.intrinsics.width, camera.intrinsics.height)

    def _delete_framebuffer(self):
        if self._framebuf is not None:
            glDeleteFramebuffers(1, [self._framebuf])
        if self._colorbuf is not None:
            glDeleteRenderbuffers(1, [self._colorbuf])
        if self._depthbuf is not None:
            glDeleteRenderbuffers(1, [self._depthbuf])

        self._framebuf = None
        self._colorbuf = None
        self._depthbuf = None
        self._framebuf_dims = (None, None)
