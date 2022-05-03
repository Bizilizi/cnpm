import fnmatch
import os
import typing as t

import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
from tqdm import tqdm


def get_all_files(match_pattern: str, path: str = ".") -> t.List[t.Tuple[str, str]]:
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, match_pattern):
            matches.append((root, filename))
            break
        break

    return matches


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Model trainer')
    # parser.add_argument('path', type=str, help='Path to traverse')
    # args = parser.parse_args()

    files = get_all_files("*.obj", "/Volumes/Brandon/ShapeNet/ShapeNetCore.v2")
    for root, file_name in tqdm(files):
        mesh = trimesh.load("root" + "/" + "file_name")
        voxels = mesh_to_voxels(mesh, 128, pad=True, sign_method="depth")

        file_name, _ = file_name.split(".")
        np.save(f"voxels", voxels)
