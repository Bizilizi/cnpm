import itertools
import math
import typing as t

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCGridRaysampler,
    RayBundle,
    look_at_view_transform,
    ray_bundle_to_ray_points,
)
from skimage import measure
from torch import nn
from torch.nn import Parameter

from src.util.volumetric_rendering.harmonic_embedding import HarmonicEmbedding
from src.util.volumetric_rendering.ray_bundle_chunk import chunk_ray_bundle
from src.util.volumetric_rendering.ray_matcher import (
    EmissionAbsorptionRaymarcherWithFragments,
)


class ColorNPMV2(pl.LightningModule):
    def __init__(
        self,
        shape_latent_num: int,
        shape_latent_dim: int,
        color_latent_dim: int,
        color_latent_num: int,
        color_suits_num: int,
        enforce_minmax: bool = True,
        positional_emb_size: int = 30,
        direction_emb_sizes: int = 30,
        *,
        model_learning_rate: float,
        latent_code_learning_rate: float,
        lambda_code_regularization: float,
        max_processing_points: int = 50000,
    ):
        super(ColorNPMV2, self).__init__()
        self.save_hyperparameters()

        # hyperparmaters
        self.lambda_code_regularization = lambda_code_regularization
        self.latent_code_learning_rate = latent_code_learning_rate
        self.model_learning_rate = model_learning_rate
        self.max_processing_points = max_processing_points

        # latent dims
        self.positional_emb_size = positional_emb_size
        self.direction_emb_sizes = direction_emb_sizes
        self.shape_latent_num = shape_latent_num
        self.color_latent_num = color_latent_num

        self.shape_latent_dim = shape_latent_dim
        self.color_latent_dim = color_latent_dim

        self.color_suits_num = color_suits_num
        self.enforce_minmax = enforce_minmax

        # volumetric rendering
        self.raymarcher = EmissionAbsorptionRaymarcherWithFragments()
        self.dens_eps = 0.05

        self._init_losses()
        self._init_layers()

    def _init_losses(self) -> None:
        # volumetric rendering
        self.vr_silhouette_loss = nn.L1Loss()
        self.vr_color_background_loss = nn.MSELoss()
        self.vr_color_object_loss = nn.MSELoss()

    def _init_layers(self) -> None:
        self.positional_emb = HarmonicEmbedding(
            n_harmonic_functions=self.positional_emb_size
        )
        self.direction_emb = HarmonicEmbedding(
            n_harmonic_functions=self.direction_emb_sizes
        )

        self.shape_latent_vectors = nn.Embedding(
            self.shape_latent_num,
            self.shape_latent_dim,
            max_norm=1.0,
        )

        self.color_latent_vectors = nn.Embedding(
            self.color_latent_num,
            self.color_latent_dim,
            max_norm=1.0,
        )

        positional_emb_shape = self.positional_emb_size * 6
        directional_emb_shape = self.direction_emb_sizes * 6

        # shape decoder
        self.shape_decoder_1 = nn.Sequential(
            nn.Linear(self.shape_latent_dim + positional_emb_shape, 512),
            nn.Softplus(beta=10),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            torch.nn.Softplus(beta=10),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.Softplus(beta=10),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Softplus(beta=10),
            nn.Dropout(0.2),
        )
        self.shape_decoder_2 = nn.Sequential(
            nn.Linear(256 + self.shape_latent_dim + positional_emb_shape, 512),
            nn.Softplus(beta=10),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.Softplus(beta=10),
            nn.Dropout(0.2),
        )
        self.density_output = nn.Sequential(
            nn.Linear(512, 512),
            nn.Softplus(beta=10),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            torch.nn.Softplus(beta=10),
        )

        # color decoder
        self.color_decoder = nn.Sequential(
            nn.Linear(
                self.color_latent_dim + directional_emb_shape + 512,
                512,
            ),
            nn.Softplus(beta=10),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Softplus(beta=10),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )

    # region FORWARD PASS
    def forward(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
        shape_emb: torch.Tensor,
        color_emb: torch.Tensor,
        with_raw_densities: bool = False,
    ) -> t.Union[
        t.Tuple[torch.Tensor, torch.Tensor],
        t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        # computes embedding
        position_emb = self.positional_emb(points)

        directions = F.normalize(directions, dim=-1)
        direction_emb = self.direction_emb(directions)

        # pass through shape encoder
        shape_inp = torch.cat([position_emb, shape_emb], dim=-1)
        x = self.shape_decoder_1(shape_inp)
        x = torch.cat([x, shape_inp], dim=-1)
        x = self.shape_decoder_2(x)

        # predict densities
        raw_densities = self.density_output(x)
        densities = 1 - torch.exp(-raw_densities)

        # predict colors
        color_inp = torch.cat([color_emb, direction_emb, x], dim=-1)
        color_out = self.color_decoder(color_inp)

        if with_raw_densities:
            return raw_densities, densities, color_out
        else:
            return densities, color_out

    def render_single(
        self,
        shape_latent_code,
        color_latent_code,
        render_size: int = 256,
        volume_extent_world: float = 2.7,
        dist: float = 2.7,
        elev: float = 0,
        azim: float = 0,
        num_batch: int = 10,
    ):
        """
        Render single image with volumetric rendering approach.
        Rendering is heavy operation, so better to turn of gradient, plus function is
        not differentiable.

        @param shape_latent_code: Latent code for rendered shape
        @param color_latent_code: Latent code for rendered color
        @param render_size: Rendered image size
        @param volume_extent_world: Distance of camera from center of coordinate
        @param elev: Elevation angle of camera view
        @param azim: Azimuth angle of camera view
        @param dist: Distance from where to make render
        @param num_batch: Number of batches for ray processing

        @return: Tuple with predicrted renders:
                    - predicted_colors_at_rays [H,W, 3]
                    - predicted_silhouettes_at_rays [H, W]
        """
        num_points_per_pay = 128
        grid_raysampler = NDCGridRaysampler(
            image_height=render_size,
            image_width=render_size,
            n_pts_per_ray=num_points_per_pay,
            min_depth=0.1,
            max_depth=volume_extent_world,
        )

        # prepare cameras
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(
            R=R,
            T=T,
            device=self.device,
        )

        # sample ray bundle
        rays_bundle = grid_raysampler(cameras=cameras)

        rays_points = ray_bundle_to_ray_points(rays_bundle)
        rays_points_world = rays_points.reshape(-1, 3)

        rays_directions = (
            rays_bundle.directions.unsqueeze(3)
            .expand(-1, -1, -1, num_points_per_pay, -1)
            .reshape(-1, 3)
        )

        rays_densities = []
        color_out = []
        for batch_points, batch_directions in zip(
            rays_points_world.chunk(num_batch), rays_directions.chunk(num_batch)
        ):
            rays_shape_latent_vectors = shape_latent_code.expand(
                batch_points.shape[0], -1
            ).to(self.device)

            rays_color_latent_vectors = color_latent_code.expand(
                batch_points.shape[0], -1
            ).to(self.device)

            densities, c_out = self.forward(
                batch_points.to(self.device),
                batch_directions.to(self.device),
                rays_shape_latent_vectors,
                rays_color_latent_vectors,
            )

            rays_densities.append(densities)
            color_out.append(c_out)

        rays_densities = torch.cat(rays_densities)
        color_out = torch.cat(color_out)

        rays_densities = rays_densities.reshape(*rays_points.shape[:-1], -1)
        rays_features = color_out.reshape(*rays_points.shape[:-1], -1)

        fragments = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=rays_bundle,
        )

        predicted_colors_at_rays = fragments.features.cpu().detach()
        predicted_silhouettes_at_rays = fragments.opacities.cpu().detach()
        # predicted_silhouettes_mask = (predicted_silhouettes_at_rays < 0.5).reshape(
        #     predicted_silhouettes_at_rays.shape[:-1]
        # )
        # predicted_colors_at_rays[predicted_silhouettes_mask] += 1

        # remove redundant dimension
        predicted_colors_at_rays = predicted_colors_at_rays.squeeze()
        predicted_silhouettes_at_rays = predicted_silhouettes_at_rays.squeeze()

        return predicted_colors_at_rays, predicted_silhouettes_at_rays

    # endregion

    # region SUPPLEMENTARY
    @staticmethod
    def compute_mesh(sdf_values):
        if 0 < sdf_values.min() or 0 > sdf_values.max():
            mesh = None
            empty = True
        else:
            try:
                mesh = measure.marching_cubes(sdf_values)
                empty = False
            except ValueError:
                mesh = None
                empty = True

        return mesh, empty

    @staticmethod
    def _extract_vertices_and_faces(
        mesh, flip_axes=False
    ) -> t.Tuple[np.ndarray, np.ndarray]:

        vertices, faces = mesh[:2]
        if flip_axes:
            vertices[:, 2] = vertices[:, 2] * -1
            vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]

        return vertices.astype(np.float32), faces.astype(np.int32)

    # endregion

    # region LOSS
    def compute_vr_loss(
        self,
        predicted_silhouettes_at_rays,
        target_silhouettes_at_rays,
        predicted_colors_at_rays,
        target_colors_at_rays,
        target_silhouettes_mask,
    ):
        """
        Computes volumetric rendering loss
        """
        predicted_colors_at_rays[~target_silhouettes_mask] += 1

        # color loss
        rendered_background_color_loss = self.vr_color_background_loss(
            target_colors_at_rays[~target_silhouettes_mask],
            predicted_colors_at_rays[~target_silhouettes_mask],
        )
        rendered_object_color_loss = self.vr_color_object_loss(
            target_colors_at_rays[target_silhouettes_mask],
            predicted_colors_at_rays[target_silhouettes_mask],
        )
        rendered_color_loss = (
            rendered_background_color_loss + rendered_object_color_loss
        )

        # silhouette loss
        rendered_silhouette_loss = self.vr_silhouette_loss(
            target_silhouettes_at_rays, predicted_silhouettes_at_rays
        )

        return rendered_color_loss, rendered_silhouette_loss

    @staticmethod
    def compute_background_loss(
        chunk_weights_at_rays,
        target_silhouettes_mask,
    ):
        """
        Computes loss over vr weights
        """
        empty_rays_weights = chunk_weights_at_rays[~target_silhouettes_mask]
        if empty_rays_weights.shape[0]:
            return empty_rays_weights.pow(2).sum(-1).mean()

        return 0

    def compute_density_loss(
        self,
        weights_at_rays,
        rays_lengths,
        rays_depth_values,
    ):
        """
        Computes density loss w.r.t target normal distribution around
        rendered depth value
        """

        target_silhouettes_mask = (rays_depth_values != -1).reshape(
            rays_depth_values.shape[:-1]
        )
        ne_lengths = rays_lengths[target_silhouettes_mask]
        ne_depth = rays_depth_values[target_silhouettes_mask]
        ne_densities = weights_at_rays[target_silhouettes_mask]

        norm_pdf = (
            1
            / (np.sqrt(2 * np.pi) * self.dens_eps)
            * torch.exp(-0.5 * (ne_lengths - ne_depth) ** 2 / self.dens_eps**2)
        )
        loss = F.mse_loss(norm_pdf, ne_densities)

        return loss

    @staticmethod
    def expected_depth_loss(weights_at_rays, rays_lengths, rays_depth_values):
        mean_at_rays = (weights_at_rays * rays_lengths).sum(dim=-1, keepdim=True)
        return F.mse_loss(rays_depth_values, mean_at_rays)

    # endregion

    # region PARAMETERS
    def freeze_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

    @property
    def shape_parameters(self) -> t.Iterator[Parameter]:
        shape_parameter = [
            self.shape_decoder_1.parameters(),
            self.shape_decoder_2.parameters(),
            self.density_output.parameters(),
        ]
        return itertools.chain(*shape_parameter)

    @property
    def colors_parameters(self) -> t.Iterator[Parameter]:
        return self.color_decoder.parameters()

    # endregion

    # region PYTORCH LIGHTNING
    def configure_optimizers(self):
        optimizer_params = []

        optimizer_params.extend(
            [
                {
                    "params": self.shape_latent_vectors.parameters(),
                    "lr": self.latent_code_learning_rate,
                },
                {"params": self.shape_parameters, "lr": self.model_learning_rate},
            ]
        )

        optimizer_params.extend(
            [
                {
                    "params": self.color_latent_vectors.parameters(),
                    "lr": self.latent_code_learning_rate,
                },
                {"params": self.colors_parameters, "lr": self.model_learning_rate},
            ]
        )

        optimizer = torch.optim.Adam(optimizer_params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

        return [optimizer], [scheduler]

    def volumetric_renderring_training_step(self, batch: t.Dict[str, t.Any]):
        # Extract values for volumetric rendering
        ray_bundle = batch["ray_bundle"]
        silhouettes_at_rays = batch["silhouettes_at_rays"]
        colors_at_rays = batch["colors_at_rays"]
        depth_at_rays = batch["depth_value_at_rays"]

        # prepare losses
        silhouette_loss = 0
        color_loss = 0
        empty_ray_loss = 0
        density_loss = 0
        exp_depth_loss = 0
        batch_size = colors_at_rays.shape[0]

        # Perform forward pass on ray points
        num_chunks = math.ceil(
            ray_bundle.lengths.shape.numel() / self.max_processing_points
        )
        for (
            is_last_chunk,
            chunk_rb,
            (chunk_target_silhouettes, chunk_target_color, chunk_target_depth),
        ) in chunk_ray_bundle(
            ray_bundle=ray_bundle,
            max_points=self.max_processing_points,
            targets=[silhouettes_at_rays, colors_at_rays, depth_at_rays],
        ):
            loss = 0

            # unwrap rays points
            rays_points = ray_bundle_to_ray_points(chunk_rb)
            rays_points_world = rays_points.reshape(-1, 3)

            # allocate color/shape codes w.r.t points number
            num_ray_points_per_shape = rays_points.shape[1:-1].numel()
            total_num_ray_points = rays_points_world.shape[0]
            num_points_per_pay = chunk_rb.lengths.shape[-1]

            # chunk_rb.directions has shape=[batch, renders, rays, 3]
            # so we extend it to number of point per ray
            rays_directions = (
                chunk_rb.directions.unsqueeze(3)
                .expand(-1, -1, -1, num_points_per_pay, -1)
                .reshape(-1, 3)
            )

            rays_shape_latent_vectors = (
                self.shape_latent_vectors(batch["shape_indices"])
                .unsqueeze(1)  # [num_shapes, 1, latent_dim]
                .expand(-1, num_ray_points_per_shape, -1)
                .reshape((total_num_ray_points, self.shape_latent_dim))
            )

            rays_color_latent_vectors = (
                self.color_latent_vectors(batch["color_indices"])
                .unsqueeze(1)
                .expand(-1, num_ray_points_per_shape, -1)
                .reshape((total_num_ray_points, self.color_latent_dim))
            )

            rays_raw_densities, rays_densities, chunk_color_out = self.forward(
                rays_points_world,
                rays_directions,
                rays_shape_latent_vectors,
                rays_color_latent_vectors,
                with_raw_densities=True,
            )

            rays_densities = rays_densities.reshape(*rays_points.shape[:-1], -1)
            rays_features = chunk_color_out.reshape(*rays_points.shape[:-1], -1)
            rays_raw_densities = rays_raw_densities.reshape(*rays_points.shape[:-1])

            fragments = self.raymarcher(
                rays_densities=rays_densities,
                rays_features=rays_features,
                ray_bundle=ray_bundle,
            )
            predicted_colors_at_rays = fragments.features
            predicted_silhouettes_at_rays = fragments.opacities
            chunk_weights_at_rays = fragments.weights

            # compute loss
            silhouettes_mask = chunk_target_silhouettes.bool().reshape(
                chunk_target_silhouettes.shape[:-1]
            )

            chunk_color_loss, chunk_silhouette_loss = self.compute_vr_loss(
                predicted_colors_at_rays=predicted_colors_at_rays,
                target_colors_at_rays=chunk_target_color,
                predicted_silhouettes_at_rays=predicted_silhouettes_at_rays,
                target_silhouettes_at_rays=chunk_target_silhouettes,
                target_silhouettes_mask=silhouettes_mask,
            )
            chunk_empty_rays_loss = self.compute_background_loss(
                chunk_weights_at_rays=chunk_weights_at_rays,
                target_silhouettes_mask=silhouettes_mask,
            )

            chunk_density_loss = self.compute_density_loss(
                weights_at_rays=rays_raw_densities,
                rays_lengths=chunk_rb.lengths,
                rays_depth_values=chunk_target_depth,
            )

            chunk_exp_depth_loss = self.expected_depth_loss(
                weights_at_rays=rays_raw_densities,
                rays_lengths=chunk_rb.lengths,
                rays_depth_values=chunk_target_depth,
            )

            loss = (
                chunk_color_loss
                + chunk_silhouette_loss
                + chunk_empty_rays_loss
                + chunk_density_loss
                + chunk_exp_depth_loss
            ) / num_chunks

            color_loss += chunk_color_loss
            silhouette_loss += chunk_silhouette_loss
            empty_ray_loss += chunk_empty_rays_loss
            density_loss += chunk_density_loss
            exp_depth_loss += chunk_exp_depth_loss

            if not is_last_chunk:
                loss.backward()

            self.log(
                "loss",
                loss,
                batch_size=batch_size,
            )

        # Log volumetric losses
        self.log(
            "vr_color_loss",
            color_loss,
            batch_size=batch_size,
        )

        self.log(
            "density_loss",
            density_loss,
            batch_size=batch_size,
        )

        self.log(
            "vr_silhouette_loss",
            silhouette_loss,
            batch_size=batch_size,
        )

        self.log(
            "empty_ray_loss",
            empty_ray_loss,
            batch_size=batch_size,
        )

        self.log(
            "exp_depth_loss",
            exp_depth_loss,
            batch_size=batch_size,
        )

        return loss

    def training_step(self, batch: t.Dict[str, t.Any], batch_idx):
        loss = self.volumetric_renderring_training_step(batch)

        # Make a scheduler step
        if self.trainer and self.trainer.is_last_batch:
            self.lr_schedulers().step()

        return loss

    def transfer_batch_to_device(
        self,
        batch: t.Dict[str, t.Any],
        device: torch.device,
        dataloader_idx: int,
    ) -> t.Dict[str, torch.Tensor]:

        batch["shape_indices"] = batch["shape_indices"].to(device)
        batch["color_indices"] = batch["color_indices"].to(device)
        batch["colors_at_rays"] = batch["colors_at_rays"].to(device)
        batch["silhouettes_at_rays"] = batch["silhouettes_at_rays"].to(device)
        batch["depth_value_at_rays"] = batch["depth_value_at_rays"].to(device)

        ray_bundle: RayBundle = batch["ray_bundle"]
        batch["ray_bundle"] = RayBundle(
            origins=ray_bundle.origins.to(device),
            directions=ray_bundle.directions.to(device),
            lengths=ray_bundle.lengths.to(device),
            xys=ray_bundle.xys.to(device),
        )

        return batch

    # endregion
