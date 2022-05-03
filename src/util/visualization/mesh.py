import typing as t
from itertools import zip_longest

import numpy as np
import trimesh
from plotly import graph_objects as go


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def visualize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: t.Optional[np.ndarray] = None,
    flip_axes: bool = False,
    tool: str = "plotly",
):
    if flip_axes:
        rot_matrix = np.array(
            [
                [-1.0000000, 0.0000000, 0.0000000],
                [0.0000000, 0.0000000, 1.0000000],
                [0.0000000, 1.0000000, 0.0000000],
            ]
        )
        vertices = vertices @ rot_matrix

    if tool == "plotly":
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    lighting=dict(ambient=0.7, roughness=0.1, diffuse=0.1),
                    colorscale="Greys",
                    intensity=z,
                    showscale=False,
                )
            ]
        )
        fig.update_layout(
            scene_aspectmode="data",
        )

        return fig
    elif tool == "trimesh":
        if normals is None:
            raise ValueError("Argument 'normal' can't be None with tool == trimesh")

        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_normals=-1 * normals
        )
        return mesh
    else:
        raise ValueError(f"No such tool: {tool}")


def visualize_meshes(
    meshes: t.List[t.Tuple[np.ndarray, np.ndarray]],
    n_columns: int = 5,
    flip_axes: bool = False,
    tool: str = "plotly",
):
    y_shift = 0
    max_length = float("-inf")
    n_columns = min(n_columns, len(meshes))

    for mesh_idx, row_mesh in enumerate(grouper(meshes, n_columns)):
        x_shift = 0
        for mesh in row_mesh:
            if mesh:
                vertices = mesh[0]

                if flip_axes:
                    vertices[:, 2] = vertices[:, 2] * -1
                    vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]

                vertices += [
                    -vertices[0:].min() + x_shift,
                    -vertices[1:].min() + y_shift,
                    0,
                ]
                x_shift = vertices[0, :].max()
                max_length = max(max_length, vertices[1, :].max())
        y_shift = max_length

    if tool == "plotly":

        data = []

        for mesh in meshes:
            vertices, faces = mesh[:2]
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

            data.append(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    lighting=dict(ambient=0.7, roughness=0.1, diffuse=0.1),
                    colorscale="Greys",
                    intensity=z,
                    showscale=False,
                )
            )

        fig = go.Figure(data=data)
        fig.update_layout(
            scene_aspectmode="data",
        )

        return fig
    elif tool == "trimesh":
        m_vertices, m_faces, m_normals = [], [], []
        for mesh in meshes:
            m_vertices.append(mesh[0])
            m_faces.append(mesh[1])
            m_faces.append(mesh[2])

        return trimesh.Trimesh(
            vertices=np.vstack(m_vertices),
            faces=np.vstack(m_faces),
            vertex_normals=-1 * np.vstack(m_normals),
        )
    else:
        raise ValueError(f"No such tool: {tool}")
