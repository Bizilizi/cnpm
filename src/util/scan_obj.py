import os
import typing as t
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments


class Renders(t.NamedTuple):
    cameras: FoVPerspectiveCameras
    images: torch.Tensor
    silhouette: torch.Tensor
    depth_maps: torch.Tensor
    elev: torch.Tensor
    azim: torch.Tensor


def _process_batch(
    elev,
    azim,
    device,
    image_size,
    mesh,
    dist: float = 2.7,
):
    # Initialize an OpenGL perspective camera that represents a batch of different
    # viewing angles.
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Place a ambient lights
    lights = AmbientLights(ambient_color=torch.tensor([[0.8, 0.8, 0.8]]), device=device)

    # Define the settings for rasterization and shading.
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=50
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured
    # Phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    # Create a batch of meshes by repeating the mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(elev.shape[0])

    target_images, fragments = renderer(meshes, cameras=cameras, lights=lights)
    depth = fragments.zbuf

    # Rasterization settings for silhouette rendering
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=50,
    )

    # Silhouette renderer
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader(),
    )

    # Render silhouette images.  The 3rd channel of the rendering output is
    # the alpha/silhouette channel
    silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)

    return target_images, silhouette_images, depth


def generate_renders(
    mesh_path: t.Union[str, Path],
    num_views: int = 40,
    image_size: int = 128,
    dist: float = 2.7,
    azimuth_range: float = 180,
    elev_range: float = 180,
    device: t.Optional[torch.device] = None,
    target_device: t.Optional[torch.device] = None,
    num_batch: int = 0,
) -> Renders:
    """
    Given mesh path produces multiple scans from different camera positions.

    **Remark**
    In case of a CUDA device this method consumes quit a lot of GPU memory. Don't forget
    to call torch.cuda.empty_cache().


    @param mesh_path: Path to obj file to be scanned
    @param num_views: Number of views to render
    @param image_size: Size of rendered image
    @param dist: Distance of camera from coordinate center. Mesh is normalized before
                 render. Keep it in mind when choose dist parameter.
    @param azimuth_range: Azimuth angle range. Camera positions would be sampled
                          along the range values [-azimuth_range, azimuth_range]
    @param elev_range: Elevation angle range. Camera positions would be sampled
                          along the range values [-elev_range, elev_range]
    @param device: Device to be used for the render procedure
    @param target_device: Device to be used to store result renders. If None, then
                          the device from the "device" parameter will be used.

    @return: Renders instance. Where:
             - "cameras" field is FoVPerspectiveCameras representing set of sampled
               cameras positions.
             - "images" is tensor of shape (num_views, image_size, image_size, 3)
             - "silhouette" is tensor of shape (num_views, image_size, image_size, 1)
             - "elev" and "azim" are tensors of shape (num_views). Each camera for
               particular render was initiated with transformation obtained by:
               look_at_view_transform(dist=2.7, elev=elev[i], azim=azim[i])
    """
    # Setup
    device = device or torch.device("cpu")
    if isinstance(mesh_path, Path):
        mesh_path = str(mesh_path)

    # Load obj file
    obj_filename = os.path.join(mesh_path)
    mesh = load_objs_as_meshes([obj_filename], device=device, create_texture_atlas=True)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
    # to its original center and scale.  Note that normalizing the target mesh,
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-(center.expand(N, 3)))
    mesh.scale_verts_((1.0 / float(scale)))

    # Get a batch of viewing angles.
    elev = torch.linspace(-elev_range, elev_range, num_views) + 180.0
    azim = torch.linspace(-azimuth_range, azimuth_range, num_views) + 180.0

    # Render the mesh from each viewing angle
    if num_batch:
        images = []
        silhouette_images = []
        depth_maps = []

        indices = torch.range(0, num_views - 1).long()
        for chunk_idx in indices.chunk(num_batch):
            rendered_images, rendered_silhouette, rendered_depth = _process_batch(
                elev=elev[chunk_idx],
                azim=azim[chunk_idx],
                device=device,
                image_size=image_size,
                mesh=mesh,
                dist=dist,
            )
            images.append(rendered_images)
            silhouette_images.append(rendered_silhouette)
            depth_maps.append(rendered_depth[..., 0])

        images = torch.cat(images)
        silhouette_images = torch.cat(silhouette_images)
        depth_maps = torch.cat(depth_maps)
    else:
        images, silhouette_images, depth_maps = _process_batch(
            elev=elev,
            azim=azim,
            device=device,
            image_size=image_size,
            mesh=mesh,
            dist=dist,
        )
        depth_maps = depth_maps[..., 0]

    # binary silhouettes
    silhouette_binary = (silhouette_images[..., 3] > 1e-4).float()

    # recreate cameras
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    if target_device:
        cameras = cameras.to(target_device)
        images = images[..., :3].to(target_device)
        silhouette_binary = silhouette_binary.to(target_device)
        depth_maps = depth_maps.to(target_device)
        elev = elev.to(target_device)
        azim = azim.to(target_device)

    render = Renders(
        cameras=cameras,
        images=images[..., :3],
        silhouette=silhouette_binary,
        depth_maps=depth_maps,
        elev=elev,
        azim=azim,
    )

    return render
