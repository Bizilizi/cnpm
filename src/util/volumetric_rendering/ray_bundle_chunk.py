import math
import typing as t

import torch
from pytorch3d.renderer import RayBundle


def chunk_ray_bundle(
    ray_bundle: RayBundle, max_points: int, targets: t.List[torch.Tensor]
) -> t.Iterable[RayBundle]:
    num_chunks = math.ceil(ray_bundle.lengths.shape.numel() / max_points)

    for i, (origins, directions, lengths, xys, *chunk_targets) in enumerate(
        zip(
            *map(
                lambda tensor: tensor.chunk(num_chunks, dim=2),
                [
                    ray_bundle.origins,
                    ray_bundle.directions,
                    ray_bundle.lengths,
                    ray_bundle.xys,
                ]
                + targets,
            )
        ),
    ):
        yield (
            i == num_chunks - 1,
            RayBundle(origins=origins, directions=directions, lengths=lengths, xys=xys),
            chunk_targets,
        )
