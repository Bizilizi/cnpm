import argparse
from configparser import ConfigParser

from pytorch_lightning import seed_everything

from src.cnpm.training.trainer import (
    construct_typed_config as cnpm_construct_typed_config,
)
from src.cnpm.training.trainer import train as train_cnpm
from src.cnpm_v2.training.trainer import (
    construct_typed_config as cnpm_v2_construct_typed_config,
)
from src.cnpm_v2.training.trainer import train as train_cnpm_v2


config_pathes = {
    "deepsdf": "./src/deepsdf/train_config.ini",
    "3depn": "./src/threedepn/train_config.ini",
    "cnpm": "./src/cnpm/train_config.ini",
    "cnpm-v2": "./src/cnpm_v2/train_config.ini",
}

config_factories = {
    "cnpm": cnpm_construct_typed_config,
    "cnpm-v2": cnpm_v2_construct_typed_config,
}

training_loops = {
    "cnpm": train_cnpm,
    "cnpm-v2": train_cnpm_v2,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer")
    parser.add_argument("--model", type=str, help="name of a model to train")
    parser.add_argument(
        "--resume_from",
        nargs="?",
        type=str,
        help="Path to checkpoint",
        default="",
    )
    parser.add_argument(
        "--seed",
        nargs="?",
        type=int,
        help="Seed for random generator",
        default=42,
    )
    parser.add_argument(
        "--run_id",
        nargs="?",
        type=str,
        help="Wandb run id to reattach",
        default="",
    )
    args = parser.parse_args()

    # make it deterministic
    seed_everything(args.seed)

    model_name = args.model
    if model_name not in ["cnpm", "cnpm-v2"]:
        raise ValueError("Unsupported models!")

    # Reading configuration from ini file
    print(f"Reading training configuration for {model_name}..")
    ini_config = ConfigParser()
    path_to_config = config_pathes[model_name]
    ini_config.read(path_to_config)

    # Unpack config to typed config class
    config = config_factories[model_name](ini_config)

    # Run training process
    print(f"Running training process for {model_name}..")
    try:
        training_loops[model_name](config, args.resume_from, args.run_id)
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
