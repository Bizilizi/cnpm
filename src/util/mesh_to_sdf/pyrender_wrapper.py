### Wrapper around the pyrender library that allows to
### 1. disable antialiasing
### 2. render a normal buffer
### This needs to be imported before pyrender or OpenGL is imported anywhere

import os
import sys

import numpy as np

# Disable antialiasing:
import OpenGL.GL

# if "pyrender" in sys.modules:
#     raise ImportError(
#         "The mesh_to_sdf package must be imported before pyrender is imported."
#     )
# if "OpenGL" in sys.modules:
#     raise ImportError(
#         "The mesh_to_sdf package must be imported before OpenGL is imported."
#     )


suppress_multisampling = False
old_gl_enable = OpenGL.GL.glEnable


def new_gl_enable(value):
    if suppress_multisampling and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)


OpenGL.GL.glEnable = new_gl_enable

old_glRenderbufferStorageMultisample = OpenGL.GL.glRenderbufferStorageMultisample


def new_glRenderbufferStorageMultisample(
    target, samples, internalformat, width, height
):
    if suppress_multisampling:
        OpenGL.GL.glRenderbufferStorage(target, internalformat, width, height)
    else:
        old_glRenderbufferStorageMultisample(
            target, samples, internalformat, width, height
        )


OpenGL.GL.glRenderbufferStorageMultisample = new_glRenderbufferStorageMultisample

import pyrender


# Render a normal buffer instead of a color buffer
class CustomShaderCache:
    def __init__(self):
        self.program = None

    def get_program(
        self, vertex_shader, fragment_shader, geometry_shader=None, defines=None
    ):
        if self.program is None:
            shaders_directory = os.path.join(os.path.dirname(__file__), "shaders")
            self.program = pyrender.shader_program.ShaderProgram(
                os.path.join(shaders_directory, "mesh.vert"),
                os.path.join(shaders_directory, "mesh.frag"),
                defines=defines,
            )
        return self.program


def render_normal_and_depth_buffers(mesh, camera, camera_transform, resolution):
    global suppress_multisampling
    suppress_multisampling = True

    # make norm render
    scene = pyrender.Scene.from_trimesh_scene(mesh)
    scene.add(camera, pose=camera_transform)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    renderer._renderer._program_cache = CustomShaderCache()

    norms, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)

    # make color render
    scene = pyrender.Scene.from_trimesh_scene(mesh)
    scene.add(camera, pose=camera_transform)
    scene.ambient_light = np.array([0.6, 0.6, 0.6, 1.0])

    color_renderer = pyrender.OffscreenRenderer(resolution, resolution)
    colors, _ = color_renderer.render(scene)

    suppress_multisampling = False

    return norms, depth, colors
