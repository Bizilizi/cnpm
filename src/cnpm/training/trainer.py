import typing as t

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from ..callbacks import LogColorsCallback, LogMeshesCallback, LogModelWightsCallback
from ..callbacks.log_renders import LogRendersCallback
from ..data.colored_shapenet import ColoredShapeNetDataModule
from ..model.color_npm import ColorNPM
from .config import TrainingConfig


def construct_typed_config(ini_config):
    config = TrainingConfig(
        **ini_config["training"],
        **ini_config["dataset"],
        **ini_config["tensorboard"],
        **ini_config["cnpm"]
    )

    return config


def train(
    config: TrainingConfig,
    resume_from_checkpoint: t.Optional[str] = None,
    run_id: t.Optional[str] = None,
):

    # Create DataModule
    datamodule = ColoredShapeNetDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_samples_point=config.num_samples_point,
        num_renders=config.num_renders,
        n_rays_per_image=config.n_rays_per_image,
        n_pts_per_ray=config.n_pts_per_ray,
        max_ray_depth=config.max_ray_depth,
        num_color_shifts=config.num_color_shifts,
        path_to_split=(
            config.path_to_overfit_split
            if config.is_overfit
            else config.path_to_train_split
        ),
        path_to_dataset=config.path_to_dataset,
        with_rays=config.with_vr_silhouette_loss or config.with_vr_color_loss,
        with_value_fields=config.with_color_fields_loss or config.with_sdf_fields_loss,
    )
    datamodule.setup()

    # Instantiate model
    model = ColorNPM(
        color_suits_num=datamodule.num_color_shifts,
        shape_latent_num=datamodule.train_dataset.shape_num,
        shape_latent_dim=config.shape_latent_code_length,
        color_latent_num=datamodule.train_dataset.color_num,
        color_latent_dim=config.color_latent_code_length,
        model_learning_rate=config.learning_rate_model,
        latent_code_learning_rate=config.learning_rate_code,
        lambda_code_regularization=config.lambda_code_regularization,
        train_shape=config.train_shape,
        train_color=config.train_color,
        with_sdf_fields_loss=config.with_sdf_fields_loss,
        with_color_fields_loss=config.with_color_fields_loss,
        with_vr_color_loss=config.with_vr_color_loss,
        with_vr_silhouette_loss=config.with_vr_silhouette_loss,
        max_processing_points=config.max_processing_points,
    )

    # Create logger
    if config.logger_type == "wandb":
        wandb.init(project="CNPM")
        logger = pl_loggers.WandbLogger(project="CNPM", log_model="all")
        logger.watch(model)
    elif config.logger_type == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(save_dir=config.logging_path)
    else:
        logger = None

    # Training
    trainer = Trainer(
        gpus=config.gpus,
        check_val_every_n_epoch=config.validate_every_n,
        default_root_dir=config.resume_ckpt_path,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=config.max_epochs,
        callbacks=[
            LogRendersCallback(log_every=config.log_renders_every),
            LogMeshesCallback(log_every=config.log_meshes_every),
            LogModelWightsCallback(log_every=config.log_model_every),
        ],
        amp_backend="native",
        resume_from_checkpoint=resume_from_checkpoint,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=datamodule)
