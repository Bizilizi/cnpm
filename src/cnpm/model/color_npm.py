import itertools
import typing as t

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch3d.renderer import (
    EmissionAbsorptionRaymarcher,
    FoVPerspectiveCameras,
    NDCGridRaysampler,
    RayBundle,
    look_at_view_transform,
    ray_bundle_to_ray_points,
)
from skimage import measure
from torch import nn
from torch.nn import Parameter
from torch.nn.utils import weight_norm

from src.util.volumetric_rendering.harmonic_embedding import HarmonicEmbedding
from src.util.volumetric_rendering.ray_bundle_chunk import chunk_ray_bundle


class ColorNPM(pl.LightningModule):
    def __init__(
        self,
        shape_latent_num: int,
        shape_latent_dim: int,
        color_latent_dim: int,
        color_latent_num: int,
        color_suits_num: int,
        enforce_minmax: bool = True,
        positional_emb_size: int = 30,
        *,
        model_learning_rate: float,
        latent_code_learning_rate: float,
        lambda_code_regularization: float,
        train_shape: bool = True,
        train_color: bool = True,
        with_sdf_fields_loss: bool = True,
        with_color_fields_loss: bool = True,
        with_vr_silhouette_loss: bool = True,
        with_vr_color_loss: bool = True,
        max_processing_points: int = 50000,
    ):
        super(ColorNPM, self).__init__()
        self.save_hyperparameters()

        # hyperparmaters
        self.lambda_code_regularization = lambda_code_regularization
        self.latent_code_learning_rate = latent_code_learning_rate
        self.model_learning_rate = model_learning_rate
        self.max_processing_points = max_processing_points

        # latent dims
        self.positional_emb_size = positional_emb_size
        self.shape_latent_num = shape_latent_num
        self.color_latent_num = color_latent_num

        self.shape_latent_dim = shape_latent_dim
        self.color_latent_dim = color_latent_dim

        self.color_suits_num = color_suits_num
        self.enforce_minmax = enforce_minmax

        # Loss configuration
        self.train_shape = train_shape
        self.train_color = train_color

        self.with_sdf_fields_loss = with_sdf_fields_loss
        self.with_color_fields_loss = with_color_fields_loss
        self.with_vr_silhouette_loss = with_vr_silhouette_loss
        self.with_vr_color_loss = with_vr_color_loss

        # volumetric rendering
        self.raymarcher = EmissionAbsorptionRaymarcher()

        self._init_losses()
        self._init_layers()

    def _init_losses(self) -> None:
        # value field
        self.shape_loss_criterion = nn.L1Loss()
        self.color_loss_criterion = nn.L1Loss()

        # volumetric rendering
        self.vr_silhouette_loss = nn.L1Loss()
        self.vr_color_loss = nn.L1Loss()

    def _init_layers(self) -> None:
        self.positional_emb = HarmonicEmbedding(
            n_harmonic_functions=self.positional_emb_size
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
        # shape decoder
        self.shape_decoder_1 = nn.Sequential(
            weight_norm(nn.Linear(self.shape_latent_dim + positional_emb_shape, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 256)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.shape_decoder_2 = nn.Sequential(
            weight_norm(
                nn.Linear(256 + self.shape_latent_dim + positional_emb_shape, 512)
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.density_output = weight_norm(nn.Linear(512, 1))
        self.shape_output_layer = weight_norm(nn.Linear(512, 1))
        self.tanh = nn.Tanh()

        # color decoder
        self.color_decoder_1 = nn.Sequential(
            weight_norm(
                nn.Linear(
                    self.color_latent_dim
                    + self.shape_latent_dim
                    + positional_emb_shape,
                    512,
                )
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 256)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.color_decoder_2 = nn.Sequential(
            weight_norm(
                nn.Linear(
                    256
                    + self.color_latent_dim
                    + self.shape_latent_dim
                    + positional_emb_shape,
                    512,
                )
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.color_output_layer = weight_norm(nn.Linear(512, 3))

    # region FORWARD PASS
    def forward(
        self,
        points: torch.Tensor,
        shape_emb: torch.Tensor,
        color_emb: torch.Tensor,
        return_density: bool = True,
        return_sdf: bool = True,
    ) -> t.Tuple[
        t.Optional[torch.Tensor], t.Optional[torch.Tensor], t.Optional[torch.Tensor]
    ]:

        # pass through shape encoder
        shape_out = densities = None
        position_emb = self.positional_emb(points)
        shape_inp = torch.cat([position_emb, shape_emb], dim=-1)

        if self.train_shape:
            # If we train shape, then backprop through shape params
            x = self.shape_decoder_1(shape_inp)
            x = torch.cat([x, shape_inp], dim=-1)
            x = self.shape_decoder_2(x)

            if return_density:
                densities = self.density_output(x)
                densities = 1 - (-densities).exp()

            if return_sdf:
                shape_out = self.shape_output_layer(x)
                shape_out = self.tanh(shape_out)

        elif self.volumetric_rendering_loss:
            # If we dont train shape, but use volumetric rendering,
            # then just predict sdf without backprop
            with torch.no_grad():
                x = self.shape_decoder_1(shape_inp)
                x = torch.cat([x, shape_inp], dim=-1)
                x = self.shape_decoder_2(x)

                if return_density:
                    densities = self.density_output(x)
                    densities = 1 - (-densities).exp()

                if return_sdf:
                    shape_out = self.shape_output_layer(x)
                    shape_out = self.tanh(shape_out)

        # pass through color encoder
        color_out = None
        if self.train_color:
            # If we train color, then backprop through color params
            color_inp = torch.cat([position_emb, shape_emb.detach(), color_emb], dim=-1)

            x = self.color_decoder_1(color_inp)
            x = torch.cat([x, color_inp], dim=-1)
            x = self.color_decoder_2(x)
            x = self.color_output_layer(x)
            color_out = torch.tanh(x)
        elif self.volumetric_rendering_loss:
            # Otherwise predict random color for volumetric rendering consistency
            color_out = torch.rand(size=(points.shape[0], 3)).to(self.device)

        return densities, shape_out, color_out

    def chunk_forward(
        self,
        points: torch.Tensor,
        shape_emb: torch.Tensor,
        color_emb: torch.Tensor,
        return_density: bool = True,
        return_sdf: bool = True,
    ) -> t.Iterable[
        t.Tuple[
            bool,
            t.Optional[torch.Tensor],
            t.Optional[torch.Tensor],
            t.Optional[torch.Tensor],
        ]
    ]:
        """
        Chunk input data into small sized pieces and perform forward operation on them.
        This is a generator function that yields data triples.

        @param points: Tensor of shape [n_points, 3]
        @param shape_emb: Tensor of shape [n_points, shape_latent_dim]
        @param color_emb: Tensor of shape [n_points, color_latent_dim]
        @param return_density: Whether or not calculate/include density to the return
        @param return_sdf: Whether or not calculate/include SDF to the return
        @return: Yields triples:
                 - is_last_chunk : True if chunk is the last one
                 - densities : Corresponding output of forward function (chunk)
                 - shape_out : Corresponding output of forward function (chunk)
                 - color_out : Corresponding output of forward function (chunk)
        """

        if points.shape[0] > self.max_processing_points:
            num_chunks = int(np.ceil(points.shape[0] / self.max_processing_points))
            for i, (points_chunk, shape_emb_chunk, color_emb_chunk) in enumerate(
                zip(
                    points.chunk(num_chunks),
                    shape_emb.chunk(num_chunks),
                    color_emb.chunk(num_chunks),
                )
            ):
                densities, chunk_shape_out, chunk_color_out = self.forward(
                    points_chunk,
                    shape_emb_chunk,
                    color_emb_chunk,
                    return_density=return_density,
                    return_sdf=return_sdf,
                )

                yield i == num_chunks - 1, densities, chunk_shape_out, chunk_color_out

        else:
            densities, shape_out, color_out = self.forward(
                points,
                shape_emb,
                color_emb,
                return_density=return_density,
                return_sdf=return_sdf,
            )

            yield True, densities, shape_out, color_out

    def infer_single(
        self,
        shape_latent_code,
        color_latent_code,
        grid_resolution,
        clamp: bool = True,
    ):
        # save old values
        old_with_color = self.train_color
        old_with_shape = self.train_shape

        self.train_color = self.train_shape = True

        # Evaluate model on grid
        x_range = y_range = z_range = np.linspace(-1.0, 1.0, grid_resolution)
        grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

        stacked = (
            torch.from_numpy(
                np.hstack(
                    (
                        grid_x[:, np.newaxis],
                        grid_y[:, np.newaxis],
                        grid_z[:, np.newaxis],
                    )
                )
            )
            .float()
            .to(self.device)
        )
        stacked_split = torch.split(stacked, 32**3, dim=0)
        sdf_values = []
        color_values = []

        for points in stacked_split:
            shape_latent_codes = shape_latent_code.unsqueeze(0).expand(
                points.shape[0], -1
            )
            color_latent_codes = color_latent_code.unsqueeze(0).expand(
                points.shape[0], -1
            )
            _, sdf, colors = self.forward(
                points, shape_latent_codes, color_latent_codes, return_density=False
            )

            if clamp:
                sdf = torch.clamp(sdf, -0.1, 0.1)

            sdf_values.append(sdf.detach().cpu())

            color_values.append(colors.detach().cpu())

        sdf_values = (
            torch.cat(sdf_values, dim=0)
            .numpy()
            .reshape((grid_resolution, grid_resolution, grid_resolution))
            .copy()
        )

        color_values = (
            torch.cat(color_values, dim=0)
            .numpy()
            .reshape((grid_resolution, grid_resolution, grid_resolution, 3))
            .copy()
        )

        self.train_shape = old_with_shape
        self.train_color = old_with_color

        return grid_x, grid_y, grid_z, sdf_values, color_values

    def render_single(
        self,
        shape_latent_code,
        color_latent_code,
        render_size: int = 256,
        volume_extent_world: float = 2.7,
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
        @param num_batch: Number of batches for ray processing

        @return: Tuple with predicrted renders:
                    - predicted_colors_at_rays [H,W, 3]
                    - predicted_silhouettes_at_rays [H, W]
        """
        grid_raysampler = NDCGridRaysampler(
            image_height=render_size,
            image_width=render_size,
            n_pts_per_ray=128,
            min_depth=0.1,
            max_depth=volume_extent_world,
        )

        R, T = look_at_view_transform(dist=3, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(
            R=R,
            T=T,
            device=self.device,
        )

        rays_bundle = grid_raysampler(cameras=cameras)
        rays_points = ray_bundle_to_ray_points(rays_bundle)

        rays_points_world = rays_points.reshape(-1, 3)

        rays_densities = []
        color_out = []
        for batch in rays_points_world.chunk(num_batch):
            rays_shape_latent_vectors = shape_latent_code.expand(batch.shape[0], -1).to(
                self.device
            )

            rays_color_latent_vectors = color_latent_code.expand(batch.shape[0], -1).to(
                self.device
            )

            densities, _, c_out = self.forward(
                batch.to(self.device),
                rays_shape_latent_vectors,
                rays_color_latent_vectors,
                return_sdf=False,
            )

            rays_densities.append(densities)
            color_out.append(c_out)

        rays_densities = torch.cat(rays_densities)
        color_out = torch.cat(color_out)

        rays_densities = rays_densities.reshape(*rays_points.shape[:-1], -1)
        rays_features = color_out.reshape(*rays_points.shape[:-1], -1)

        rendered_images_silhouettes = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=rays_bundle,
        )

        (
            predicted_colors_at_rays,
            predicted_silhouettes_at_rays,
        ) = rendered_images_silhouettes.split([3, 1], dim=-1)

        predicted_colors_at_rays = predicted_colors_at_rays / 255
        predicted_colors_at_rays = predicted_colors_at_rays.cpu().detach()

        predicted_silhouettes_at_rays = predicted_silhouettes_at_rays[..., 0]
        predicted_silhouettes_at_rays = predicted_silhouettes_at_rays.clamp(0, 1)

        # Filter values that dont contribute to 90% of sum
        flattened = predicted_silhouettes_at_rays.reshape(-1)
        max_idx = torch.argsort(flattened, descending=True)

        cumsum = torch.cumsum(flattened[max_idx], 0)
        a_sum = cumsum < cumsum[-1] * 0.80
        shifted_a_sum = a_sum.roll(1)
        shifted_a_sum[0] = True

        div_idx = torch.where(torch.logical_xor(a_sum, shifted_a_sum))[0]
        flattened[max_idx[div_idx:]] = 0

        # Leave only filtered values for silhouette and color
        predicted_silhouettes_at_rays = flattened.reshape(
            predicted_silhouettes_at_rays.shape
        )
        predicted_silhouettes_at_rays = predicted_silhouettes_at_rays.cpu().detach()

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
    ):
        """
        Computes volumetric rendering loss
        """
        rendered_color_loss = self.vr_color_loss(
            target_colors_at_rays, predicted_colors_at_rays
        )
        rendered_silhouette_loss = self.vr_silhouette_loss(
            target_silhouettes_at_rays, predicted_silhouettes_at_rays
        )

        return rendered_color_loss, rendered_silhouette_loss

    def compute_shape_loss(
        self,
        predicted_sdf,
        target_sdf,
        batch_shape_latent_vectors,
    ):
        """
        Computes shape loss between predicted and target sdf fields.
        """
        if self.enforce_minmax:
            predicted_sdf = torch.clamp(predicted_sdf, -0.1, 0.1)

        shape_loss = self.shape_loss_criterion(predicted_sdf, target_sdf)

        # regularize latent codes
        if self.current_epoch > 100:
            shape_loss += (
                torch.mean(torch.norm(batch_shape_latent_vectors, dim=1))
                * self.lambda_code_regularization
            )

        return shape_loss

    def compute_color_loss(
        self,
        target_sdf,
        predicted_color,
        target_colors,
        batch_color_latent_vectors,
    ):
        """
        Computes color loss between target and predicted color fields.
        """
        if self.enforce_minmax:
            close_points = (target_sdf <= 0.1) & (target_sdf >= -0.1)
            close_points = close_points.flatten()
            predicted_color = predicted_color[close_points]
            target_colors = target_colors[close_points]

        color_loss = self.color_loss_criterion(predicted_color, target_colors)

        # regularize latent codes
        if self.current_epoch > 100:
            color_loss += (
                torch.mean(torch.norm(batch_color_latent_vectors, dim=1))
                * self.lambda_code_regularization
            )

        return color_loss

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
            self.shape_output_layer.parameters(),
            self.density_output.parameters(),
        ]
        return itertools.chain(*shape_parameter)

    @property
    def colors_parameters(self) -> t.Iterator[Parameter]:
        colors_parameter = [
            self.color_decoder_1.parameters(),
            self.color_decoder_2.parameters(),
            self.color_output_layer.parameters(),
        ]
        return itertools.chain(*colors_parameter)

    # endregion

    # region PYTORCH LIGHTNING
    def configure_optimizers(self):
        optimizer_params = []

        if self.train_shape:
            optimizer_params.extend(
                [
                    {
                        "params": self.shape_latent_vectors.parameters(),
                        "lr": self.latent_code_learning_rate,
                    },
                    {"params": self.shape_parameters, "lr": self.model_learning_rate},
                ]
            )
        if self.train_color:
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

    def value_fields_training_step(self, batch: t.Dict[str, t.Any]):
        # Calculate number of samples per batch
        # (= number of shapes in batch * number of points per shape)
        batch_size = batch["points"].shape[0]
        num_points_per_batch = batch_size * batch["points"].shape[1]

        # Get shape latent codes corresponding to batch shapes
        batch_shape_latent_vectors = (
            self.shape_latent_vectors(batch["shape_indices"])
            .unsqueeze(1)
            .expand(-1, batch["points"].shape[1], -1)
            .reshape((num_points_per_batch, self.shape_latent_dim))
        )

        # Get color latent codes corresponding to batch shapes
        batch_color_latent_vectors = (
            self.color_latent_vectors(batch["color_indices"])
            .unsqueeze(1)
            .expand(-1, batch["points"].shape[1], -1)
            .reshape((num_points_per_batch, self.color_latent_dim))
        )

        # Reshape points and sdf for forward pass
        points = batch["points"].reshape((num_points_per_batch, 3))
        sdf = batch["sdf"].reshape((num_points_per_batch, 1))
        colors = batch["color"].reshape((num_points_per_batch, 3))
        if self.enforce_minmax:
            sdf = torch.clamp(sdf, -0.1, 0.1)

        # compute losses
        color_loss = 0
        shape_loss = 0

        # Perform chunked forward pass on value fields
        for is_last_chunk, _, predicted_sdf, predicted_color in self.chunk_forward(
            points,
            batch_shape_latent_vectors,
            batch_color_latent_vectors,
        ):
            loss = 0
            if self.train_color and self.with_color_fields_loss:
                chunk_color_loss = self.compute_color_loss(
                    target_sdf=sdf,
                    predicted_color=predicted_color,
                    target_colors=colors,
                    batch_color_latent_vectors=batch_color_latent_vectors,
                )
                loss += chunk_color_loss
                color_loss += chunk_color_loss

            if self.train_shape and self.with_sdf_fields_loss:
                chunk_shape_loss = self.compute_shape_loss(
                    predicted_sdf=predicted_sdf,
                    target_sdf=sdf,
                    batch_shape_latent_vectors=batch_shape_latent_vectors,
                )
                loss += chunk_shape_loss
                shape_loss += chunk_shape_loss

            if not is_last_chunk:
                loss.backward()

        if self.train_shape and self.with_sdf_fields_loss:
            # Log shape loss
            self.log(
                "sdf_field_loss",
                shape_loss,
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
            )

        if self.train_color and self.with_color_fields_loss:
            # Log color loss
            self.log(
                "color_field_loss",
                color_loss,
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
            )

        return loss

    def volumetric_renderring_training_step(self, batch: t.Dict[str, t.Any]):
        # Extract values for volumetric rendering
        ray_bundle = batch["ray_bundle"]
        silhouettes_at_rays = batch["silhouettes_at_rays"]
        colors_at_rays = batch["colors_at_rays"]

        # prepare losses
        silhouette_loss = 0
        color_loss = 0
        batch_size = colors_at_rays.shape[0]

        # Perform forward pass on ray points
        for (
            is_last_chunk,
            chunk_rb,
            (chunk_target_silhouettes, chunk_target_color),
        ) in chunk_ray_bundle(
            ray_bundle=ray_bundle,
            max_points=self.max_processing_points,
            targets=[silhouettes_at_rays, colors_at_rays],
        ):
            loss = 0
            rays_points = ray_bundle_to_ray_points(chunk_rb)
            rays_points_world = rays_points.reshape(-1, 3)

            num_ray_points_per_shape = rays_points.shape[1:-1].numel()
            num_ray_points = rays_points_world.shape[0]

            rays_shape_latent_vectors = (
                self.shape_latent_vectors(batch["shape_indices"])
                .unsqueeze(1)  # [num_shapes, 1, latent_dim]
                .expand(-1, num_ray_points_per_shape, -1)
                .reshape((num_ray_points, self.shape_latent_dim))
            )

            rays_color_latent_vectors = (
                self.color_latent_vectors(batch["color_indices"])
                .unsqueeze(1)
                .expand(-1, num_ray_points_per_shape, -1)
                .reshape((num_ray_points, self.color_latent_dim))
            )

            rays_densities, _, chunk_color_out = self.forward(
                rays_points_world,
                rays_shape_latent_vectors,
                rays_color_latent_vectors,
                return_sdf=False,
            )
            rays_densities = rays_densities.reshape(*rays_points.shape[:-1], -1)

            rays_features = chunk_color_out.reshape(*rays_points.shape[:-1], -1)
            rendered_images_silhouettes = self.raymarcher(
                rays_densities=rays_densities,
                rays_features=rays_features,
                ray_bundle=ray_bundle,
            )
            (
                predicted_colors_at_rays,
                predicted_silhouettes_at_rays,
            ) = rendered_images_silhouettes.split([3, 1], dim=-1)

            chunk_color_loss, chunk_silhouette_loss = self.compute_vr_loss(
                predicted_colors_at_rays=predicted_colors_at_rays,
                target_colors_at_rays=chunk_target_color,
                predicted_silhouettes_at_rays=predicted_silhouettes_at_rays,
                target_silhouettes_at_rays=chunk_target_silhouettes,
            )

            if self.train_color and self.with_vr_color_loss:
                loss += chunk_color_loss
                color_loss += chunk_color_loss

            if self.train_shape and self.with_vr_silhouette_loss:
                loss += chunk_silhouette_loss
                silhouette_loss += chunk_silhouette_loss

            if not is_last_chunk:
                loss.backward()

        # Log volumetric losses
        if self.train_color and self.with_vr_color_loss:
            self.log(
                "vr_color_loss",
                color_loss,
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
            )

        if self.train_shape and self.with_vr_silhouette_loss:
            self.log(
                "vr_silhouette_loss",
                silhouette_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        return loss

    def training_step(self, batch: t.Dict[str, t.Any], batch_idx):
        loss = 0

        if self.with_vr_color_loss or self.with_vr_silhouette_loss:
            loss += self.volumetric_renderring_training_step(batch)

        if self.with_sdf_fields_loss or self.with_color_fields_loss:
            loss += self.value_fields_training_step(batch)

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

        if "points" in batch and "sdf" in batch and "color" in batch:
            batch["points"] = batch["points"].to(device)
            batch["sdf"] = batch["sdf"].to(device)
            batch["color"] = batch["color"].to(device)

        if (
            "colors_at_rays" in batch
            and "silhouettes_at_rays" in batch
            and "ray_bundle" in batch
        ):
            batch["colors_at_rays"] = batch["colors_at_rays"].to(device)
            batch["silhouettes_at_rays"] = batch["silhouettes_at_rays"].to(device)

            ray_bundle: RayBundle = batch["ray_bundle"]
            batch["ray_bundle"] = RayBundle(
                origins=ray_bundle.origins.to(device),
                directions=ray_bundle.directions.to(device),
                lengths=ray_bundle.lengths.to(device),
                xys=ray_bundle.xys.to(device),
            )

        return batch

    # endregion
