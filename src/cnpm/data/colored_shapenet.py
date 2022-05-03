import typing as t
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MonteCarloRaysampler,
    look_at_view_transform,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import adjust_hue

from src.util.volumetric_rendering.sample_images_at_mc_locs import (
    sample_images_at_mc_locs,
)


class ColoredShapeNet(Dataset):
    """
    Dataset for loading colored ShapeNet dataset
    """

    def __init__(
        self,
        dataset_path: str,
        num_sample_points: int,
        num_renders: int,
        n_rays_per_image: int = 750,
        n_pts_per_ray: int = 128,
        max_ray_depth: float = 3.0,
        num_color_shifts: int = 20,
        split: t.Optional[str] = None,
        with_rays: bool = True,
        with_value_fields: bool = True,
    ):
        """
        :@param num_sample_points: number of points to sample for sdf values per shape
        :@param num_shifts: number of colorized shifts
        :@param num_renders: number of renders per batch
        :@param split: one of 'train', 'val' or 'overfit' - for training,
                      validation or overfitting split
        :@param with_rays: If True each sample will contain ray points
        """
        super().__init__()

        self.num_renders = num_renders
        self.dataset_path = Path(dataset_path)
        self.num_sample_points = num_sample_points
        self.items = Path(split).read_text().splitlines()
        self.num_color_shifts = num_color_shifts
        self.shape_num = len(self.items)
        self.color_num = len(self.items) * self.num_color_shifts
        self.raysampler = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=n_rays_per_image,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=0.1,
            max_depth=max_ray_depth,
        )

        # Sample type parametrization
        self.with_rays = with_rays
        self.with_value_fields = with_value_fields

    def __getitem__(self, index):
        """
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of sdf data corresponding to the shape. In particular, this dictionary has keys
                 "name", shape_identifier of the shape
                 "indices": index parameter
                 "points": a num_sample_points x 3  pytorch float32 tensor containing sampled point coordinates
                 "sdf", a num_sample_points x 1 pytorch float32 tensor containing sdf values for the sampled points
        """

        # get shape_id at index
        shape_id = index // self.num_color_shifts
        shape_name = self.items[shape_id]

        sample = {
            "name": shape_name,  # identifier of the shape
            "shape_indices": shape_id,  # shape index parameter
            "color_indices": index,  # color index parameter
        }

        if self.with_value_fields:
            # read points and their sdf values from disk
            samples = self.get_sdf_samples(shape_name)

            points = samples[:, :3]
            sdf = samples[:, 3]
            color = self.hue_shift(samples[:, 4:], index, self.num_color_shifts)

            # add value fields to sample
            sample.update(
                {
                    "points": points,  # tensor with shape num_sample_points x 3
                    "sdf": sdf,  # tensor with shape num_sample_points x 1
                    "color": color,  # tensor with shape num_sample_points x 3
                }
            )

        if self.with_rays:
            cameras, images, silhouettes, _, _ = self.get_renders(shape_name)
            ray_bundle = self.raysampler(cameras=cameras)

            # compute silhouette
            silhouettes_at_rays = sample_images_at_mc_locs(
                silhouettes[..., None], ray_bundle.xys
            )
            silhouette_mask = silhouettes_at_rays.squeeze().bool()

            # compute colors after hue shift
            colors_at_rays = sample_images_at_mc_locs(images, ray_bundle.xys)
            colors_at_rays[silhouette_mask] = self.hue_shift(
                colors_at_rays[silhouette_mask],
                index,
                self.num_color_shifts,
                norm=False,
            )
            colors_at_rays = colors_at_rays * 2 - 1  # rescale it to -1, 1

            sample.update(
                {
                    # ↓ n_rays_per_image rays sampled from renders
                    "ray_bundle": ray_bundle,
                    # ↓ [num_renders, n_rays_per_image, n_pts_per_ray, 3]
                    "colors_at_rays": colors_at_rays,
                    # ↓ [num_renders, n_rays_per_image, n_pts_per_ray, 1]
                    "silhouettes_at_rays": silhouettes_at_rays,
                }
            )

        return sample

    def __len__(self):
        """
        :return: length of the dataset
        """

        return self.color_num

    def _extract_tensors(self, shape_id: str) -> t.Tuple[torch.Tensor, torch.Tensor]:
        # read data
        path_to_npz = self.dataset_path / shape_id / "models" / "sdf_color.pt"
        pt = torch.load(path_to_npz)

        pos_tensor = pt["positive_sdf"]
        neg_tensor = pt["negative_sdf"]

        return pos_tensor, neg_tensor

    @staticmethod
    def hue_shift(
        colors: torch.Tensor, index: int, num_color_shifts: int, norm: bool = False
    ) -> torch.Tensor:
        color_idx = index % num_color_shifts
        if norm:
            colors = colors / 255

        # # transform color
        hue_factor = color_idx / num_color_shifts
        hue_factor = (-1) * hue_factor // 0.5 * hue_factor % 0.5

        colors = (
            adjust_hue(
                colors[None].permute(2, 0, 1),
                hue_factor,
            )
            .squeeze()
            .permute(1, 0)
        )

        # set new color
        if norm:
            colors = colors * 255

        return colors

    def get_renders(
        self,
        shape_id: str,
        dist: float = 2.7,
        renders_idx: t.Optional[t.List[int]] = None,
    ) -> t.Tuple[
        FoVPerspectiveCameras, torch.Tensor, torch.Tensor, t.List[float], t.List[float]
    ]:
        """
        Return renders for specific shape_id

        @param dist:
        @param shape_id: Id of shape for which to extract renders. Random sample if None
        @param renders_idx: Indices of renders
        @return: Tuple of elements:
                    - Cameras: FoVPerspectiveCameras for each render
                    - Images: Tensor of size (N, H, W, C)
                    - silhouettes: Tensor of size (N, H, W, C)
                    - Elev: Elevation angle values
                    - Azim: Azimuth angle values
        """
        path_to_scans = self.dataset_path / str(shape_id) / "scans"
        scans = [f.name for f in path_to_scans.iterdir() if f.is_file()]

        if renders_idx is None:
            scans = np.random.choice(scans, size=self.num_renders, replace=False)
        else:
            scans = [scans[i] for i in renders_idx]

        elev = []
        azim = []
        images = []
        silhouettes = []

        for file_name in scans:
            path_to_scan = path_to_scans / file_name
            scan = torch.load(path_to_scan)

            # unwrap scan data
            elev.append(scan["elev"])
            azim.append(scan["azim"])
            images.append(scan["image"][None])
            silhouettes.append(scan["silhouette"][None])

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(R=R, T=T)
        images = torch.cat(images)
        silhouettes = torch.cat(silhouettes)

        return cameras, images, silhouettes, elev, azim

    def get_sdf_samples(self, shape_id: str):
        """
        Method for reading an sdf file; the SDF file for a shape contains
        a number of points, along with their sdf values and color

        :@param path_to_npz: path to npz file
        :@param color_idx: index of color suit
        :@return: a pytorch float32 torch tensor of shape (num_sample_points, 4)
                  with each row being [x, y, z, sdf_value, color at xyz]
        """
        pos_tensor, neg_tensor = self._extract_tensors(shape_id)

        N = self.num_sample_points // 2
        idx = np.random.choice(
            pos_tensor.shape[0],
            size=N,
        )
        samples_pos = pos_tensor[idx]

        idx = np.random.choice(
            neg_tensor.shape[0],
            size=self.num_sample_points - N,
        )
        samples_neg = neg_tensor[idx]

        return torch.cat([samples_pos, samples_neg], dim=0)

    def get_mesh(self, shape_id):
        """
        Loading a mesh from the shape with identifier

        :param shape_id: shape identifier for ShapeNet object
        :return: trimesh object representing the mesh
        """
        return trimesh.load(
            self.dataset_path / shape_id / "models" / "mesh.obj", force="mesh"
        )

    def get_scene(self, shape_id):
        """
        Loading a trimesh Scene instance from the shape with identifier

        :param shape_id: shape identifier for ShapeNet object
        :return: trimesh.Scene object representing the mesh
        """
        return trimesh.load(
            self.dataset_path / shape_id / "models" / "model_normalized.obj"
        )

    def get_all_sdf_samples(self, shape_id: str):
        """
        Loading all points and their sdf values

        :@param shape_id: shape identifier for ShapeNet object
        :@param color_id: color identifier, e.g 0,1,3
        :@return: three torch float32 tensors:
                    - Nx3 tensor containing point coordinates
                    - Nx1 tensor containing their sdf values
                    - Nx3 tensor containing point color

        """
        pos_tensor, neg_tensor = self._extract_tensors(shape_id)

        samples = torch.cat([pos_tensor, neg_tensor], 0)

        points = samples[:, :3]
        sdf = samples[:, 3]
        colors = samples[:, 4:]

        return points, sdf, colors


class EmptyDataset(Dataset):
    """
    Mock dataset for DeppSDF empty validation step
    """

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return {
            "name": torch.tensor([1]),
            "indices": torch.tensor([1]),
            "points": torch.tensor([1]),
            "sdf": torch.tensor([1]),
            "color": torch.tensor([1]),
        }

    def __len__(self):
        return 10


class ColoredShapeNetDataModule(pl.LightningDataModule):
    train_dataset: ColoredShapeNet
    val_dataset: EmptyDataset

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        *,
        num_renders: int = 2,
        n_rays_per_image: int = 750,
        n_pts_per_ray: int = 128,
        max_ray_depth: float = 3.0,
        num_color_shifts: int = 20,
        num_samples_point: int,
        path_to_split: str,
        path_to_dataset: str,
        with_rays: bool = True,
        with_value_fields: bool = True,
    ):
        super().__init__()

        self.num_renders = num_renders
        self.max_ray_depth = max_ray_depth
        self.n_pts_per_ray = n_pts_per_ray
        self.n_rays_per_image = n_rays_per_image

        self.num_samples_point = num_samples_point
        self.path_to_dataset = path_to_dataset
        self.path_to_split = path_to_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_color_shifts = num_color_shifts

        self.with_value_fields = with_value_fields
        self.with_rays = with_rays

    def setup(self, stage: t.Optional[str] = None) -> None:
        self.train_dataset = ColoredShapeNet(
            dataset_path=self.path_to_dataset,
            num_renders=self.num_renders,
            num_sample_points=self.num_samples_point,
            num_color_shifts=self.num_color_shifts,
            split=self.path_to_split,
            with_rays=self.with_rays,
            with_value_fields=self.with_value_fields,
            n_rays_per_image=self.n_rays_per_image,
            n_pts_per_ray=self.n_pts_per_ray,
            max_ray_depth=self.max_ray_depth,
        )

        self.val_dataset = EmptyDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=1, shuffle=False)
