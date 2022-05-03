import argparse
import os
from configparser import ConfigParser
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import tqdm, trange

from src.cnpm.data import ColoredShapeNet
from src.cnpm.model import ColorNPM
from src.cnpm.scripts.optimization.initialization import (
    mean_based_initialization,
    prefect_initialization,
)
from src.cnpm.scripts.optimization.optimize import optimize
from src.cnpm.training import TrainingConfig
from src.cnpm.training.trainer import (
    construct_typed_config as cnpm_construct_typed_config,
)
from src.util.mesh_to_sdf import sample_sdf_near_surface

os.environ["PYOPENGL_PLATFORM"] = "egl"

try:
    set_start_method("spawn", force=True)
except Exception:
    ...


def add_arguments(parser):
    parser.add_argument("--model", type=str, help="name of a model to train")
    parser.add_argument(
        "--resume_from",
        nargs="?",
        type=str,
        help="path to resume model",
        default="",
    )
    parser.add_argument(
        "--work_path",
        nargs="?",
        type=str,
        help="path to working dir",
        default="./",
    )
    parser.add_argument(
        "--num_steps",
        nargs="?",
        type=int,
        help="number of steps per optimisation procedure",
        default=3000,
    )

    parser.add_argument(
        "--num_samples",
        nargs="?",
        type=int,
        help="number of samples for optimisation procedure",
        default=None,
    )

    parser.add_argument(
        "--device",
        nargs="?",
        type=str,
        help="cuda device num",
        default=None,
    )

    parser.add_argument(
        "--num_threads",
        nargs="?",
        type=int,
        help="number of threads to use",
        default=1,
    )

    parser.add_argument(
        "--init_type",
        nargs="?",
        type=str,
        help="Latent code initialization type",
        default="mean",
    )

    parser.add_argument(
        "--path_to_config",
        nargs="?",
        type=str,
        help="Path to dataset",
        default="mean",
    )


def optimization_iteration(
    sample_idx: int,
    sample_id: str,
    num_color_shifts: int,
    device_name: str,
    num_steps: int,
    working_dir: Path,
    resume_from: str,
    config: TrainingConfig,
    color_suits_num: int = 1,
    initialization_type: str = "mean",
):
    device = torch.device(device_name)
    model = ColorNPM.load_from_checkpoint(
        resume_from,
        shape_latent_num=1147,
        color_latent_num=1147,
        shape_latent_dim=config.shape_latent_code_length,
        color_latent_dim=config.color_latent_code_length,
        color_suits_num=color_suits_num,
        model_learning_rate=config.learning_rate_model,
        latent_code_learning_rate=config.learning_rate_code,
        lambda_code_regularization=config.lambda_code_regularization,
        with_shape=False,
    ).to(device)

    # define dataset

    model.eval()
    model.freeze_parameters()
    model.train_color = model.train_shape = True

    mesh = trimesh.load(f"data/sofas/{sample_id}/models/model_normalized.obj")
    (query_points, query_points_sdf, query_points_colors,) = sample_sdf_near_surface(
        mesh,
        scan_count=1,
        number_of_points=50000,
    )
    query_points_colors = torch.from_numpy(query_points_colors).float()
    query_points_colors = ColoredShapeNet.hue_shift(
        query_points_colors, sample_idx, num_color_shifts
    )
    batch = dict(
        points=torch.from_numpy(query_points[None]).float(),
        sdf=torch.from_numpy(query_points_sdf[None]).float(),
        color=query_points_colors[None],
        shape_indices=torch.tensor([sample_idx], dtype=torch.float32),
        color_indices=torch.tensor([sample_idx], dtype=torch.float32),
        name=[sample_id],
    )
    batch = model.transfer_batch_to_device(batch, model.device, 1)

    batch_size = batch["points"].shape[0]

    if initialization_type == "mean":
        (
            shape_latent_vectors_variable,
            color_latent_vectors_variable,
        ) = mean_based_initialization(batch_size, model)
    elif initialization_type == "perfect":
        (
            shape_latent_vectors_variable,
            color_latent_vectors_variable,
        ) = prefect_initialization(batch, model)
    else:
        raise ValueError(f"Unsupported initialization type: {initialization_type}")

    shapes, colors, losses, gradients = optimize(
        model,
        shape_latent_vectors_variable,
        color_latent_vectors_variable,
        batch,
        steps=num_steps,
    )

    # save results
    torch.save(
        dict(
            losses=losses,
            gradients=gradients,
            shapes=shapes,
            colors=colors,
        ),
        str(working_dir / f"{batch['name'][0]}.pt"),
    )

    del shapes
    del gradients
    del losses
    del colors
    del model
    del batch
    del color_latent_vectors_variable
    del shape_latent_vectors_variable


def uwrap_args_optimization_iteration(args):
    return optimization_iteration(**args)


def main(
    device_name: str,
    resume_from: str,
    num_samples: int,
    num_steps: int,
    work_path: str,
    num_threads: int,
    color_suits_num: int = 1,
    initialization_type: str = "mean",
):
    # define model
    ini_config = ConfigParser()
    ini_config.read("src/cnpm/train_config.ini")
    config = cnpm_construct_typed_config(ini_config)

    # define dataset
    if initialization_type == "mean":
        val_dataset = ColoredShapeNet(
            config.path_to_dataset,
            50000,
            color_suits_num,
            config.path_to_validation_split,
        )
    elif initialization_type == "perfect":
        val_dataset = ColoredShapeNet(
            config.path_to_dataset, 50000, color_suits_num, config.path_to_train_split
        )
    else:
        raise ValueError(f"Unsupported initialization type: {initialization_type}")

    if num_samples:
        samples_idx = np.random.choice(np.arange(len(val_dataset)), size=num_samples)
    else:
        samples_idx = np.arange(len(val_dataset))

    # run optimisation procedures
    working_dir = Path(work_path) / "multi_view" / initialization_type
    working_dir.mkdir(parents=True, exist_ok=True)

    mm_args = [
        dict(
            sample_idx=sample_id,
            sample_id=val_dataset.items[sample_id // color_suits_num],
            num_color_shifts=color_suits_num,
            device_name=device_name,
            num_steps=num_steps,
            working_dir=working_dir,
            resume_from=resume_from,
            config=config,
            initialization_type=initialization_type,
        )
        for sample_id in samples_idx
    ]
    if num_threads > 1:
        with Pool(num_threads) as pool:
            list(
                tqdm(
                    pool.imap_unordered(uwrap_args_optimization_iteration, mm_args),
                    total=len(mm_args),
                )
            )
    else:
        for arguments in tqdm(mm_args):
            uwrap_args_optimization_iteration(arguments)


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser(
        description="Creates dataframe with optimization history"
    )
    add_arguments(parser)
    args = parser.parse_args()

    # define model
    main(
        device_name=args.device,
        resume_from=args.resume_from,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        work_path=args.work_path,
        num_threads=args.num_threads,
        initialization_type=args.init_type,
    )
