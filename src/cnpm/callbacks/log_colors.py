import typing as t
from io import StringIO

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from trimesh.exchange.obj import export_obj

from ..model.color_npm import ColorNPM


class LogColorsCallback(Callback):
    NUM_BOARDS = 5

    def __init__(self, log_every=5):
        super().__init__()
        self.state = {"epochs": 0}
        self.log_every = log_every

    def _log_meshes(self, logger, meshes: t.List[trimesh.Trimesh]):

        if isinstance(logger, WandbLogger):
            wandb.log(
                {
                    f"colored_points_{i}": wandb.Object3D(
                        StringIO(export_obj(t_mesh)),
                        file_type="obj",
                        caption=f"Colored Object #{i}",
                    )
                    for i, t_mesh in enumerate(meshes)
                }
            )

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        model: ColorNPM,
        unused=None,
    ):
        self.state["epochs"] += 1
        if not model.train_color:
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
                shape_latent_code = shape_latent_vectors[i, :]
                color_latent_code = color_latent_vectors[i, :]

                # compute sdf values on grid
                with torch.no_grad():
                    x, y, z, sdf_values, _ = model.infer_single(
                        shape_latent_code,
                        color_latent_code,
                        128,
                    )

                # apply marching cube algorithm
                mesh, empty = model.compute_mesh(sdf_values)

                if not empty:
                    # rescale vertices
                    vertices, faces, norms = mesh[0], mesh[1], mesh[2]
                    vertices /= 128 / 2
                    vertices -= np.array([1, 1, 1])
                    vertices = torch.tensor(vertices.copy()).to(model.device)

                    # compute color on vertices
                    shape_latent_codes = shape_latent_code.unsqueeze(0).expand(
                        vertices.shape[0], -1
                    )
                    color_latent_codes = color_latent_code.unsqueeze(0).expand(
                        vertices.shape[0], -1
                    )

                    with torch.no_grad():
                        vertices_colors = []
                        for _, _, _, color_out in model.chunk_forward(
                            vertices,
                            shape_latent_codes,
                            color_latent_codes,
                            return_density=False,
                            return_sdf=False,
                        ):
                            vertices_colors.append(color_out)

                    vertices_colors = torch.cat(vertices_colors).detach().cpu()
                    vertices = vertices.detach().cpu()

                    mesh = trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces,
                        vertex_normals=-1 * norms,
                        vertex_colors=vertices_colors,
                    )
                    meshes.append(mesh)

            # Check whether we have one logger or multiple
            # and log to all loggers we have
            if isinstance(trainer.logger, LoggerCollection):
                for logger in trainer.logger:
                    self._log_meshes(logger, meshes)
            else:
                self._log_meshes(trainer.logger, meshes)

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()
