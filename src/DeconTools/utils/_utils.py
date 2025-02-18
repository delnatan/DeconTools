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


def clear_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # MPS doesn't have an explicit cache clearing mechanism
        # but we can try to force garbage collection
        import gc

        gc.collect()
        torch.mps.empty_cache()  # Added in newer PyTorch versions
