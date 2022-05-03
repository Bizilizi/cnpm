import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LoggerCollection, WandbLogger

from src.util.visualization import visualize_sdf

from ..model.color_npm import ColorNPM


class LogVolumesCallback(Callback):
    NUM_BOARDS = 5

    def __init__(self, log_every=5):
        super().__init__()
        self.state = {"epochs": 0}
        self.log_every = log_every

    def _log_volumes(self, sdf_values, logger, step):
        if isinstance(logger, WandbLogger):
            wandb.log(
                {
                    f"volume_{i}": wandb.Plotly(visualize_sdf(x, y, z, sdf_value))
                    for i, (x, y, z, sdf_value) in enumerate(sdf_values)
                }
            )

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        model: ColorNPM,
        unused=None,
    ):
        self.state["epochs"] += 1
        if not model.train_shape:
            return

        if self.state["epochs"] % self.log_every == 0:
            model.eval()
            num_boards = min(self.NUM_BOARDS, model.shape_latent_num)

            # Compute meshes
            shape_latent_vectors = model.shape_latent_vectors(
                torch.LongTensor(list(range(num_boards))).to(model.device)
            )
            color_latent_vectors = model.color_latent_vectors(
                torch.LongTensor(list(range(num_boards))).to(model.device)
            )

            sdf_values = []
            for i in range(num_boards):
                with torch.no_grad():
                    x, y, z, sdf_value, _ = model.infer_single(
                        shape_latent_vectors[i, :],
                        color_latent_vectors[i, :],
                        64,
                    )

                sdf_values.append((x, y, z, sdf_value.flatten()))

            if isinstance(trainer.logger, LoggerCollection):
                for logger in trainer.logger:
                    self._log_volumes(sdf_values, logger, step=model.global_step)
            else:
                self._log_volumes(sdf_values, trainer.logger, step=model.global_step)

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()
