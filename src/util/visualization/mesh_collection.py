import numpy as np
import pyrender
import trimesh
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )
    return nodes


def mesh_to_pic(mesh, renderer=None, resolution=512):
    renderer = renderer or pyrender.OffscreenRenderer(resolution, resolution)

    mesh = mesh.copy()
    mesh.apply_scale(1 / mesh.scale)

    camera_rotation = np.eye(4)
    camera_rotation[:3, :3] = (
        Rotation.from_euler("y", 180, degrees=True).as_matrix()
        @ Rotation.from_euler("x", -20, degrees=True).as_matrix()
    )

    camera_translation = np.eye(4)
    camera_translation[:3, 3] = np.array([0, 0, 1])

    camera_pose = camera_rotation @ camera_translation
    camera = pyrender.PerspectiveCamera(yfov=1.04, aspectRatio=1.0, znear=0.001, zfar=3)

    trimesh_scene = trimesh.Scene(geometry=mesh)
    trimesh_scene.rezero()

    scene = pyrender.Scene.from_trimesh_scene(trimesh_scene)
    scene.add(camera, pose=camera_pose)

    for n in create_raymond_lights():
        scene.add_node(n, scene.main_camera_node)

    color, depth = renderer.render(scene)

    return color


def meshes_to_gif(meshes, fps=25, resolution=512):
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    image_buffer = []

    for mesh in tqdm(meshes, desc="Visualizing meshes", leave=False):
        color = mesh_to_pic(mesh, renderer, resolution)
        image_buffer.append(color)

    clip = ImageSequenceClip(image_buffer, fps=fps)
    return clip
