import typing as t

import numpy as np
import plotly.graph_objects as go


def visualize_colored_points(
    x,
    y,
    z,
    sdf_values,
    color_values,
    *,
    distance=0.01,
    sample_size=20000,
    return_scatter=False
) -> t.Union[go.Figure, go.Scatter3d]:
    """
    Return plotly figure with colorized point near surface. Each point sampled randomly.
    Size of sample and condition on sdf value can be controlled
    by distance and sample_size parameters.


    @param x: X coordinate of a grid
    @param y: Y coordinate of a grid
    @param z: Z coordinate of a grid
    @param sdf_values: Values of signed distance function evaluated at (x,y,z) point
    @param color_values: Color values at (x,y,z) point
    @param distance: Threshold value.
                    All drawn point will have  -distance < sdf < distance
    @param sample_size: Number of points to draw
    @param return_scatter: If true returns go.Scatter3d obj instead of go.Figure

    @return go.Figure instance
    """

    sdf_values = sdf_values.flatten()
    color_values = color_values.reshape(-1, 3)

    close_points_mask = (sdf_values <= distance) & (sdf_values >= -distance)

    x = x[close_points_mask]
    y = y[close_points_mask]
    z = z[close_points_mask]
    close_points_colors = color_values[close_points_mask]

    sub_idx = np.random.choice(np.arange(x.shape[0]), size=sample_size)

    x = x[sub_idx]
    y = y[sub_idx]
    z = z[sub_idx]
    close_points_colors = close_points_colors[sub_idx]

    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=2, color=close_points_colors, opacity=1),
    )

    if return_scatter:
        return scatter

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene_aspectmode="data",
    )

    return fig
