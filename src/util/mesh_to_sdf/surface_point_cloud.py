import logging

import trimesh

from .scan import Scan, get_camera_transform_looking_at_origin
from .utils import (
    check_voxels,
    get_raster_points,
    sample_uniform_points_in_box,
    sample_uniform_points_in_sphere,
)

logging.getLogger("trimesh").setLevel(9000)
import math

import numpy as np
import pyrender
from sklearn.neighbors import KDTree


class BadMeshException(Exception):
    pass


class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None, colors=None, scans=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals
        self.scans = scans
        self.colors = colors

        self.kd_tree = KDTree(points)

    def get_random_surface_points(self, count, use_scans=True):
        if use_scans:
            indices = np.random.choice(self.points.shape[0], count)
            return self.points[indices, :]
        else:
            return self.mesh.sample(count)

    def get_sdf_and_color(
        self,
        query_points,
        use_depth_buffer=False,
        sample_count=11,
        return_gradients=False,
    ):
        if use_depth_buffer:
            distances, indices = self.kd_tree.query(query_points)
            distances = distances.astype(np.float32).reshape(-1)
            inside = ~self.is_outside(query_points)
            distances[inside] *= -1

            if return_gradients:
                gradients = query_points - self.points[indices[:, 0]]
                gradients[inside] *= -1

        else:
            distances, indices = self.kd_tree.query(query_points, k=sample_count)
            distances = distances.astype(np.float32)

            closest_points = self.points[indices]
            direction_from_surface = query_points[:, np.newaxis, :] - closest_points
            inside = (
                np.einsum("ijk,ijk->ij", direction_from_surface, self.normals[indices])
                < 0
            )
            inside = np.sum(inside, axis=1) > sample_count * 0.5
            distances = distances[:, 0]
            distances[inside] *= -1

            if return_gradients:
                gradients = direction_from_surface[:, 0]
                gradients[inside] *= -1

        if return_gradients:
            near_surface = (
                np.abs(distances) < math.sqrt(0.0025**2 * 3) * 3
            )  # 3D 2-norm stdev * 3
            gradients = np.where(
                near_surface[:, np.newaxis], self.normals[indices[:, 0]], gradients
            )
            gradients /= np.linalg.norm(gradients, axis=1)[:, np.newaxis]
            return distances, gradients
        else:
            return distances

    def get_sdf_in_batches(
        self,
        query_points,
        use_depth_buffer=False,
        sample_count=11,
        batch_size=1000000,
        return_gradients=False,
    ):
        if query_points.shape[0] <= batch_size:
            return self.get_sdf_and_color(
                query_points,
                use_depth_buffer=use_depth_buffer,
                sample_count=sample_count,
                return_gradients=return_gradients,
            )

        n_batches = int(math.ceil(query_points.shape[0] / batch_size))
        batches = [
            self.get_sdf_and_color(
                points,
                use_depth_buffer=use_depth_buffer,
                sample_count=sample_count,
                return_gradients=return_gradients,
            )
            for points in np.array_split(query_points, n_batches)
        ]
        if return_gradients:
            distances = np.concatenate([batch[0] for batch in batches])
            gradients = np.concatenate([batch[1] for batch in batches])
            return distances, gradients
        else:
            return np.concatenate(batches)  # distances

    def get_voxels(
        self,
        voxel_resolution,
        use_depth_buffer=False,
        sample_count=11,
        pad=False,
        check_result=False,
        return_gradients=False,
    ):
        result = self.get_sdf_in_batches(
            get_raster_points(voxel_resolution),
            use_depth_buffer,
            sample_count,
            return_gradients=return_gradients,
        )
        if not return_gradients:
            sdf = result
        else:
            sdf, gradients = result
            voxel_gradients = np.reshape(
                gradients, (voxel_resolution, voxel_resolution, voxel_resolution, 3)
            )

        voxels = sdf.reshape((voxel_resolution, voxel_resolution, voxel_resolution))

        if check_result and not check_voxels(voxels):
            raise BadMeshException()

        if pad:
            voxels = np.pad(voxels, 1, mode="constant", constant_values=1)
            if return_gradients:
                voxel_gradients = np.pad(
                    voxel_gradients, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge"
                )

        if return_gradients:
            return voxels, voxel_gradients
        else:
            return voxels

    def sample_sdf_near_surface(
        self,
        number_of_points=500000,
        use_scans=True,
        sign_method="normal",
        normal_sample_count=11,
        min_size=0,
        return_gradients=False,
    ):
        query_points = []
        surface_sample_count = int(number_of_points * 47 / 50) // 2
        surface_points = self.get_random_surface_points(
            surface_sample_count, use_scans=use_scans
        )
        query_points.append(
            surface_points
            + np.random.normal(scale=0.05, size=(surface_sample_count, 3))
        )
        query_points.append(
            surface_points
            + np.random.normal(scale=0.005, size=(surface_sample_count, 3))
        )
        query_points = np.concatenate(query_points).astype(np.float32)

        query_points_sdf = self.get_sdf_in_batches(
            query_points,
            use_depth_buffer=sign_method != "normal",
            sample_count=normal_sample_count,
            return_gradients=return_gradients,
        )

        # sparse points
        sparse_points_sample_count = number_of_points - surface_points.shape[0] * 2
        sparse_points = sample_uniform_points_in_box(sparse_points_sample_count)
        sparse_points = np.array(sparse_points)

        sparse_points_sdf = self.get_sdf_in_batches(
            sparse_points, use_depth_buffer=True, return_gradients=return_gradients
        )

        sdf = np.concatenate([query_points_sdf, sparse_points_sdf])
        points = np.concatenate([query_points, sparse_points])

        if return_gradients:
            sdf, gradients = sdf

        if min_size > 0:
            model_size = (
                np.count_nonzero(sdf[-sparse_points_sample_count:] < 0)
                / sparse_points_sample_count
            )
            if model_size < min_size:
                raise BadMeshException()

        _, color_indices = self.kd_tree.query(points, k=1)
        colors = self.colors[color_indices.flatten()]

        if return_gradients:
            return points, sdf, colors, gradients
        else:
            return points, sdf, colors

    def show(self):
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_points(self.points, normals=self.normals))
        pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

    def is_outside(self, points):
        result = None
        for scan in self.scans:
            if result is None:
                result = scan.is_visible(points)
            else:
                result = np.logical_or(result, scan.is_visible(points))
        return result


def get_equidistant_camera_angles(count):
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        if count == 1:
            theta = 0
        else:
            theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)

        yield phi, theta


def create_from_scans(
    mesh, bounding_radius=1, scan_count=100, scan_resolution=400, calculate_normals=True
):
    scans = []

    for phi, theta in get_equidistant_camera_angles(scan_count):
        camera_transform = get_camera_transform_looking_at_origin(
            phi, theta, camera_distance=2 * bounding_radius
        )
        scans.append(
            Scan(
                mesh,
                camera_transform=camera_transform,
                resolution=scan_resolution,
                calculate_normals=calculate_normals,
                fov=1.0472,
                z_near=bounding_radius * 1,
                z_far=bounding_radius * 3,
            )
        )

    return SurfacePointCloud(
        mesh,
        points=np.concatenate([scan.points for scan in scans], axis=0),
        normals=np.concatenate([scan.normals for scan in scans], axis=0)
        if calculate_normals
        else None,
        colors=np.concatenate([scan.colors for scan in scans], axis=0),
        scans=scans,
    )


def sample_from_mesh(mesh, sample_point_count=10000000, calculate_normals=True):
    if calculate_normals:
        points, face_indices = mesh.sample(sample_point_count, return_index=True)
        normals = mesh.face_normals[face_indices]
    else:
        points = mesh.sample(sample_point_count, return_index=False)

    return SurfacePointCloud(
        mesh, points=points, normals=normals if calculate_normals else None, scans=None
    )
