import typing as t

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    # Tensorboard
    logging_path: str

    # General configuration
    gpus: str
    is_overfit: bool
    batch_size: int
    resume_ckpt_path: str
    max_epochs: int
    validate_every_n: int
    num_workers: int
    accumulate_grad_batches: int
    max_processing_points: int
    log_renders_every: int
    log_meshes_every: int
    log_model_every: int
    logger_type: str

    # Dataset configuration
    path_to_train_split: str
    path_to_overfit_split: str
    path_to_validation_split: str
    path_to_dataset: str
    num_renders: int
    n_rays_per_image: int
    n_pts_per_ray: int
    max_ray_depth: float
    num_color_shifts: int

    # ColorNPM
    num_samples_point: int
    shape_latent_code_length: int
    color_latent_code_length: int

    learning_rate_model: float
    learning_rate_code: float

    lambda_code_regularization: float
    enforce_minmax: bool

    # training process
    train_color: bool
    train_shape: bool
    with_sdf_fields_loss: bool
    with_color_fields_loss: bool
    with_vr_silhouette_loss: bool
    with_vr_color_loss: bool
