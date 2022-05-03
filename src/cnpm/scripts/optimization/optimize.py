import torch

from src.cnpm.model import ColorNPM


def optimize(model: ColorNPM, shape_ig, color_ig, batch, steps=3000):
    def loss_function(x, y):
        batch_size = batch["points"].shape[0]
        num_points_per_batch = batch_size * batch["points"].shape[1]

        # Get shape latent codes corresponding to batch shapes
        batch_shape_latent_vectors = (
            x.unsqueeze(1)
            .expand(-1, batch["points"].shape[1], -1)
            .reshape((num_points_per_batch, model.shape_latent_dim))
        )

        # Get color latent codes corresponding to batch shapes
        batch_color_latent_vectors = (
            y.unsqueeze(1)
            .expand(-1, batch["points"].shape[1], -1)
            .reshape((num_points_per_batch, model.color_latent_dim))
        )

        points = batch["points"].reshape((num_points_per_batch, 3))
        sdf = batch["sdf"].reshape((num_points_per_batch, 1))
        colors = batch["color"].reshape((num_points_per_batch, 3))

        if model.enforce_minmax:
            sdf = torch.clamp(sdf, -0.1, 0.1)

        # Perform forward pass
        predicted_sdf, predicted_color = model.forward(
            points,
            batch_shape_latent_vectors,
            batch_color_latent_vectors,
        )

        loss = 0
        if model.train_shape:
            shape_loss = model.compute_shape_loss(
                predicted_sdf=predicted_sdf,
                target_sdf=sdf,
                batch_shape_latent_vectors=batch_shape_latent_vectors,
            )
            loss += shape_loss

        if model.train_color:
            color_loss = model.compute_color_loss(
                target_sdf=sdf,
                predicted_color=predicted_color,
                target_colors=colors,
                batch_color_latent_vectors=batch_color_latent_vectors,
            )
            loss += color_loss

        return loss

    losses = []
    gradients = []
    shapes = []
    colors = []

    optim = torch.optim.Adam([shape_ig, color_ig], lr=0.0001)

    for _ in range(steps):
        loss = loss_function(shape_ig, color_ig)
        loss.backward()
        optim.step()

        losses.append(loss.item())
        gradients.append(
            (
                shape_ig.grad.data.cpu().numpy().squeeze(),
                color_ig.grad.data.cpu().numpy().squeeze(),
            )
        )

        shape_ig.grad.data.zero_()
        color_ig.grad.data.zero_()

        shapes.append(shape_ig.data.cpu().numpy().squeeze())
        colors.append(color_ig.data.cpu().numpy().squeeze())

    return [shapes, colors, losses, gradients]
