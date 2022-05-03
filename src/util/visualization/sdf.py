import numpy as np
from plotly import graph_objects as go


def visualize_sdf(
    x: np.array,
    y: np.array,
    z: np.array,
    sdf: np.array,
    *,
    as_point_cloud: bool = False,
    isomin: float = -0.1,
    isomax: float = 0.1,
    surface_count: int = 5,
):
    if as_point_cloud:
        fig = go.Figure(
            data=go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=5,
                    color=sdf,
                    colorscale="Viridis",
                    opacity=0.2,
                    colorbar=dict(thickness=20),
                ),
            )
        )
        fig.update_layout(
            scene_aspectmode="data",
        )
    else:
        fig = go.Figure(
            data=go.Volume(
                x=x,
                y=y,
                z=z,
                value=sdf,
                isomin=isomin,
                isomax=isomax,
                opacity=0.1,
                surface_count=surface_count,
            )
        )
        fig.update_layout(
            scene_aspectmode="data",
        )
    return fig
