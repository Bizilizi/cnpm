from io import StringIO

import pytorch_lightning as pl
import torch
import trimesh
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger, WandbLogger
from trimesh.exchange.obj import export_obj

from ..model.color_npm import ColorNPM


class LogMeshesCallback(Callback):
    NUM_BOARDS = 5

    def __init__(self, log_every=5):
        super().__init__()
        self.state = {"epochs": 0}
        self.log_every = log_every

    def _log_mesh(self, logger, meshes, model):

        if isinstance(logger, WandbLogger):

            t_meshes = []
            for mesh, empty in meshes:
                if empty:
                    continue

                vertices, faces, norms = mesh[0], mesh[1], mesh[2]
                t_mesh = trimesh.Trimesh(
                    vertices=vertices, faces=faces, vertex_normals=norms
                )

                t_meshes.append(t_mesh)

            wandb.log(
                {
                    f"reconstructions_{i}": wandb.Object3D(
                        StringIO(export_obj(t_mesh)),
                        file_type="obj",
                        caption=f"Object #{i}",
                    )
                    for i, t_mesh in enumerate(t_meshes)
                },
            )
        elif isinstance(logger, TensorBoardLogger):
            for mesh, empty in meshes:
                if empty:
                    continue

                vertices, faces, norms = mesh[0], mesh[1], mesh[2]

                tensorboard = model.logger.experiment
                tensorboard.add_mesh(
                    "reconstructions",
                    vertices=vertices[None],
                    faces=faces[None],
                    global_step=model.global_step,
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

            # Compute meshes
            model.eval()
            num_boards = min(self.NUM_BOARDS, model.shape_latent_num)

            shape_latent_vectors = model.shape_latent_vectors(
                torch.LongTensor(list(range(num_boards))).to(model.device)
            )
            color_latent_vectors = model.color_latent_vectors(
                torch.LongTensor(list(range(num_boards))).to(model.device)
            )

            meshes = []
            for i in range(num_boards):
                with torch.no_grad():
                    _, _, _, sdf_values, _ = model.infer_single(
                        shape_latent_vectors[i, :],
                        color_latent_vectors[i, :],
                        128,
                    )
                mesh, empty = model.compute_mesh(sdf_values)
                meshes.append((mesh, empty))

            # Check whether we have one logger or multiple
            # and log to all loggers we have
            if isinstance(trainer.logger, LoggerCollection):
                for logger in trainer.logger:
                    self._log_mesh(logger, meshes, model)
            else:
                self._log_mesh(trainer.logger, meshes, model)

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()
