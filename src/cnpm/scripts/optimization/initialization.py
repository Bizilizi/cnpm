import typing as t

import torch
from torch.autograd import Variable

from src.cnpm.model import ColorNPM


def mean_based_initialization(
    batch_size: int, model: ColorNPM
) -> t.Tuple[Variable, Variable]:
    # Get latent codes corresponding to batch shapes
    # expand so that we have an appropriate latent vector per sdf sample
    shape_latent_vectors = torch.randn(
        (batch_size, model.shape_latent_dim), requires_grad=True
    ).to(model.device)
    shape_latent_vectors.data += 0.0065
    shape_latent_vectors.data *= 0.0545
    shape_latent_vectors_variable = Variable(shape_latent_vectors, requires_grad=True)

    color_latent_vectors = torch.randn(
        (batch_size, model.color_latent_dim), requires_grad=True
    ).to(model.device)
    color_latent_vectors.data += 0.0059
    color_latent_vectors.data *= 0.0623
    color_latent_vectors_variable = Variable(
        color_latent_vectors,
        requires_grad=True,
    )

    return shape_latent_vectors_variable, color_latent_vectors_variable


def prefect_initialization(
    batch: t.Dict[str, t.Any],
    model: ColorNPM,
):
    shape_idx = batch["shape_indices"][0]
    shape_latent_vectors = model.shape_latent_vectors(
        torch.tensor([shape_idx]).int().to(model.device),
    ).to(model.device)

    color_idx = batch["color_indices"][0]
    color_latent_vectors = model.color_latent_vectors(
        torch.tensor([color_idx]).int().to(model.device),
    ).to(model.device)

    shape_latent_vectors_variable = Variable(shape_latent_vectors, requires_grad=True)
    color_latent_vectors_variable = Variable(
        color_latent_vectors,
        requires_grad=True,
    )

    return shape_latent_vectors_variable, color_latent_vectors_variable
