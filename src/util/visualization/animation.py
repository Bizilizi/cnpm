import typing as t

import numpy as np
import plotly.graph_objects as go
import pyrender
import trimesh

from src.util.mesh_to_sdf.scan import get_camera_transform_looking_at_origin


def animate_frames(
    frames: t.List[go.Frame], title=None, as_widget: bool = False
) -> go.Figure:
    """
    Create figure for frames animation

    :returns Figure object
    """
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 10},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "left",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    for i in range(len(frames)):
        slider_step = {
            "args": [
                [f"frame_{i}"],
                {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            "label": i,
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    updatemenu_dict = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                            "transition": {
                                "duration": 300,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    if as_widget:
        figure = go.FigureWidget(
            data=[frames[0].data[0]],
            layout=go.Layout(
                title=title,
                scene_aspectmode="data",
                updatemenus=updatemenu_dict,
                sliders=[sliders_dict],
            ),
            frames=frames,
        )
    else:
        figure = go.Figure(
            data=[frames[0].data[0]],
            layout=go.Layout(
                title=title,
                scene_aspectmode="data",
                updatemenus=updatemenu_dict,
                sliders=[sliders_dict],
            ),
            frames=frames,
        )

    return figure


def animate_meshes(
    meshes: t.List[trimesh.Trimesh],
    fov=1,
    z_near=1,
    z_far=3,
    phi=2.4,
    theta=0,
    bounding_radius=1,
    resolution=400,
):
    camera = pyrender.PerspectiveCamera(
        yfov=fov, aspectRatio=1.0, znear=z_near, zfar=z_far
    )
    camera_transform = get_camera_transform_looking_at_origin(
        phi, theta, camera_distance=2 * bounding_radius
    )

    for mesh in meshes:
        scene = pyrender.Scene.from_trimesh_scene(mesh)
        scene.add(camera, pose=camera_transform)
        scene.ambient_light = np.array([0.6, 0.6, 0.6, 1.0])

        color_renderer = pyrender.OffscreenRenderer(resolution, resolution)
        colors, _ = color_renderer.render(scene)
