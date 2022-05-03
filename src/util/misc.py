import numpy as np
import torch


def remove_nans(tensor, is_torch=True):
    if is_torch:
        tensor_nan = torch.isnan(tensor[:, 3])
        return tensor[~tensor_nan, :]

    tensor_nan = np.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]
