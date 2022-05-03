import argparse
from configparser import ConfigParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from torch.multiprocessing import Process, Queue, set_start_method
from tqdm import tqdm, trange

from src.cnpm.data import ColoredShapeNet
from src.cnpm.model.color_npm import ColorNPM
from src.cnpm.scripts.optimization.initialization import mean_based_initialization
from src.cnpm.scripts.optimization.optimize import optimize
from src.cnpm.training import TrainingConfig
from src.cnpm.training.trainer import (
    construct_typed_config as cnpm_construct_typed_config,
)

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
        "--grad_path",
        nargs="?",
        type=str,
        help="path to gradient csv",
        default=None,
    )
    parser.add_argument(
        "--loss_path",
        nargs="?",
        type=str,
        help="path to loss csv",
        default=None,
    )
    parser.add_argument(
        "--color_var_path",
        nargs="?",
        type=str,
        help="path to color variabel csv",
        default=None,
    )
    parser.add_argument(
        "--shape_var_path",
        nargs="?",
        type=str,
        help="path to shape variable csv",
        default=None,
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


def make_it_batch(sample):
    """
    Extends dimensions for sample and making it batch shaped

    @param sample: Data sample
    @return: Batch
    """
    sample["name"] = [sample["name"]]
    sample["shape_indices"] = torch.tensor([sample["shape_indices"]])
    sample["color_indices"] = torch.tensor([sample["color_indices"]])
    sample["points"] = sample["points"][None, :]
    sample["sdf"] = sample["sdf"][None, :]
    sample["color"] = sample["color"][None, :]

    return sample


def bytes_to_mib(b):
    return b / 1.049e6


def get_device_mem_usage(device):
    """
    return statistics for cuda memory usage

    @param device: Cuda device
    @return: total, reserved, allocated memory in MiB
    """
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)

    return bytes_to_mib(t), bytes_to_mib(r), bytes_to_mib(a)


def uwrap_args(args):
    return optimization_iteration(*args)


def optimization_iteration(
    sample_id: int,
    device_name: str,
    num_steps: int,
    working_dir: Path,
    resume_from: str,
    config: TrainingConfig,
    color_suits_num: int = 1,
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
    val_dataset = ColoredShapeNet(
        config.path_to_dataset, 50000, color_suits_num, config.path_to_validation_split
    )

    model.eval()
    model.freeze_parameters()
    model.train_color = model.train_shape = True
    model.share_memory()

    batch = make_it_batch(val_dataset[sample_id])
    batch = model.transfer_batch_to_device(batch, model.device, 1)

    batch_size = batch["points"].shape[0]

    (
        shape_latent_vectors_variable,
        color_latent_vectors_variable,
    ) = mean_based_initialization(batch_size, model)

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


def main(
    device_name: str,
    resume_from: str,
    num_samples: int,
    num_steps: int,
    work_path: str,
    num_threads: int,
):
    # define model
    ini_config = ConfigParser()
    path_to_config = "src/cnpm/train_config.ini"
    ini_config.read(path_to_config)
    config = cnpm_construct_typed_config(ini_config)

    # define dataset
    val_dataset = ColoredShapeNet(
        config.path_to_dataset, 50000, 1, config.path_to_validation_split
    )

    if num_samples:
        samples_idx = np.random.choice(np.arange(len(val_dataset)), size=num_samples)
    else:
        samples_idx = np.arange(len(val_dataset))

    # run optimisation procedures
    working_dir = Path(work_path) / "cloud_point"
    working_dir.mkdir(parents=True, exist_ok=True)

    with Pool(num_threads) as pool:
        mm_args = [
            (
                sample_id,
                device_name,
                num_steps,
                working_dir,
                resume_from,
                config,
            )
            for sample_id in samples_idx
        ]

        list(tqdm(pool.imap_unordered(uwrap_args, mm_args), total=len(mm_args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates dataframe with optimization history"
    )
    add_arguments(parser)
    args = parser.parse_args()

    main(
        device_name=args.device,
        resume_from=args.resume_from,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        work_path=args.work_path,
        num_threads=args.num_threads,
    )
