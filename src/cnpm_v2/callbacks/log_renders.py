import typing as t

import PIL
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from torchvision import transforms
from torchvision.utils import make_grid

from ..model.color_npm import ColorNPMV2


class LogRendersCallback(Callback):
    NUM_BOARDS = 5

    def __init__(self, log_every=5):
        super().__init__()
        self.state = {"epochs": 0}
        self.log_every = log_every

    def _log_images(
        self,
        model: ColorNPMV2,
        logger,
        silhouettes: t.List["PIL.Image"],
        images: t.List["PIL.Image"],
        target_silhouettes: t.List["PIL.Image"],
        target_images: t.List["PIL.Image"],
    ):

        if isinstance(logger, WandbLogger):
            wandb.log(
                {
                    f"vr_silhouette_{i}": wandb.Image(
                        silhouette,
                        caption=f"Rendered silhouette #{i}",
                    )
                    for i, silhouette in enumerate(silhouettes)
                }
            )
            wandb.log(
                {
                    f"vr_target_silhouette_{i}": wandb.Image(
                        silhouette,
                        caption=f"Target silhouette #{i}",
                    )
                    for i, silhouette in enumerate(target_silhouettes)
                }
            )
            wandb.log(
                {
                    f"vr_image_{i}": wandb.Image(
                        image,
                        caption=f"Rendered image #{i}",
                    )
                    for i, image in enumerate(images)
                }
            )
            wandb.log(
                {
                    f"vr_target_image_{i}": wandb.Image(
                        image,
                        caption=f"Target image #{i}",
                    )
                    for i, image in enumerate(target_images)
                }
            )

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        model: ColorNPMV2,
        unused=None,
    ):

        self.state["epochs"] += 1
        if self.state["epochs"] % self.log_every == 0:

            # Compute meshes
            model.eval()
            to_pil_transform = transforms.ToPILImage()
            train_dataset = trainer.datamodule.train_dataset
            num_boards = min(self.NUM_BOARDS, model.shape_latent_num)

            shape_latent_vectors = model.shape_latent_vectors(
                torch.LongTensor(list(range(num_boards))).to(model.device)
            )
            color_latent_vectors = model.color_latent_vectors(
                torch.LongTensor(list(range(num_boards))).to(model.device)
            )

            silhouettes = []
            target_silhouettes = []
            images = []
            target_images = []

            for i in range(num_boards):
                shape_latent_code = shape_latent_vectors[i, :]
                color_latent_code = color_latent_vectors[i, :]

                # get shape_id at index
                shape_id = i // train_dataset.num_color_shifts
                shape_name = train_dataset.items[shape_id]

                # get target renders
                (
                    _,
                    target_sample_images,
                    target_sample_silhouettes,
                    _,
                    target_elev,
                    target_azim,
                    dists,
                ) = train_dataset.get_renders(
                    shape_name, renders_idx=list(torch.linspace(0, 99, 16).long())
                )

                sample_images = []
                sample_silhouettes = []

                # Render images and silhouette
                with torch.no_grad():
                    for elev, azim, dist in zip(target_elev, target_azim, dists):
                        (
                            predicted_colors_at_rays,
                            predicted_silhouettes_at_rays,
                        ) = model.render_single(
                            shape_latent_code,
                            color_latent_code,
                            render_size=256,
                            num_batch=20,
                            elev=elev,
                            azim=azim,
                            dist=dist,
                        )

                        sample_silhouettes.append(predicted_silhouettes_at_rays[None])
                        sample_images.append(predicted_colors_at_rays.permute(2, 0, 1))

                    # rendered grid
                    images.append(to_pil_transform(make_grid(sample_images, nrow=4)))
                    silhouettes.append(
                        to_pil_transform(make_grid(sample_silhouettes, nrow=4))
                    )

                    # target grid
                    target_sample_images = target_sample_images.permute(0, 3, 1, 2)
                    target_sample_silhouettes = target_sample_silhouettes[:, None, :]
                    target_images.append(
                        to_pil_transform(make_grid(target_sample_images, nrow=4))
                    )
                    target_silhouettes.append(
                        to_pil_transform(make_grid(target_sample_silhouettes, nrow=4))
                    )

            # Check whether we have one logger or multiple
            # and log to all loggers we have
            if isinstance(trainer.logger, LoggerCollection):
                for logger in trainer.logger:
                    self._log_images(
                        model,
                        logger,
                        silhouettes,
                        images,
                        target_silhouettes,
                        target_images,
                    )
            else:
                self._log_images(
                    model,
                    trainer.logger,
                    silhouettes,
                    images,
                    target_silhouettes,
                    target_images,
                )

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()
