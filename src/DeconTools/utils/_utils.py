from typing import List

import numpy as np


def move_to_torch_device(
    list_of_arrays: List[np.ndarray], target_device: str = "mps"
):
    import torch

    dev = torch.device(target_device)

    listarrs = []
    for arr in list_of_arrays:
        arr = arr.astype(np.float32)
        listarrs.append(torch.from_numpy(arr).to(dev))

    return listarrs
