import typing as t

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LoggerCollection, WandbLogger

from ..model.color_npm import ColorNPM


class LogModelWightsCallback(Callback):
    def __init__(
        self,
        log_every=5,
        checkpoint_path: str = "checkpoints",
        model_prefix: str = "model",
        model_description: t.Optional[str] = None,
    ):
        super().__init__()
        self.state = {"epochs": 0}
        self.log_every = log_every
        self.checkpoint_path = checkpoint_path
        self.model_prefix = model_prefix
        self.model_description = model_description

    def save_model_weights(self, logger, trainer: "pl.Trainer"):
        model_ckpt = (
            f"{self.checkpoint_path}/{self.model_prefix}-{self.state['epochs']}.ckpt"
        )
        trainer.save_checkpoint(model_ckpt)

        # log model to W&B
        if isinstance(logger, WandbLogger):
            artifact = wandb.Artifact(
                "model", type="model", description=self.model_description
            )
            artifact.add_file(model_ckpt)

            logger.experiment.log_artifact(artifact)

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        model: ColorNPM,
        unused=None,
    ):
        self.state["epochs"] += 1

        if self.state["epochs"] % self.log_every == 0:
            # Check whether we have one logger or multiple
            # and log to all loggers we have
            if isinstance(trainer.logger, LoggerCollection):
                for logger in trainer.logger:
                    self.save_model_weights(logger, trainer)
            else:
                self.save_model_weights(trainer.logger, trainer)

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()
