import numpy as np

from .constants import *
from .shader_program import ShaderProgramCache
from .texture import TextureCache

from OpenGL.GL import *

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

        self._objects = set()

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

    def _standard_forward_pass(self, scene, render_flags):
        if render_flags & RenderFlags.OFFSCREEN:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._framebuf)
        else:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

        # Add objects to context
        added_objects = set()
        for obj_name in scene.objects:
            obj = scene.objects[obj_name]
            if obj not in self._objects:
                obj._add_to_context()
            added_objects.add(obj)
        objs_to_remove = set()
        for obj in self._objects:
            if obj not in added_objects:
                objs_to_remove.add(obj)
        for obj in objs_to_remove:
            self._objects.remove(obj)
        self._objects = added_objects

        # Clear viewport
        self._init_standard_render(scene, render_flags)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up camera matrices
        cam_pose = scene.get_pose(scene.camera_name)
        V = scene.camera.V(cam_pose)
        P = scene.camera.P

        # Now, render each object
        prior_program = None
        for name in self._sorted_object_names(scene):
            obj = scene.objects[name]
            # Skip the object if it's not visible
            if not obj.is_visible:
                continue
            # TODO SET GL BLEND MODE

            # First, get the appropriate program
            program = self._get_program(obj) # TODO
            program.bind()

            # Next, bind the object and its materials
            obj._bind()
            # Bind material properties unless using a depth-only render.
            if not (render_flags & RenderFlags.DEPTH_ONLY):
                self._bind_object_material(name, obj, program)

            # Bind M matrix
            M = scene.get_pose(name)
            program.set_uniform('M', M)

            # First, set V and P matrices
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
            width, height = self._framebuf_dims[0], self._framebuf_dims[1]
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._framebuf)
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

        # TODO BETTER SORTING METHOD
        transparent_names.sort(
            key=lambda n : -np.linalg.norm(scene.get_pose(n)[:3,3] - cam_loc)
        )
        solid_names.sort(
            key=lambda n : -np.linalg.norm(scene.get_pose(n)[:3,3] - cam_loc)
        )

        return solid_names + transparent_names

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
