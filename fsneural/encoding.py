"""Convert env observations (numpy dicts) into batched torch tensors.

Keeping this separate from the env means the env stays framework-agnostic
(pure numpy) and the model only ever sees tensors.
"""

import numpy as np
import torch


def obs_to_tensors(obs, device="cpu"):
    """Single observation dict -> dict of (1, ...) tensors."""
    return {
        "units_self": torch.from_numpy(obs["units_self"]).unsqueeze(0).to(device),
        "units_enemy": torch.from_numpy(obs["units_enemy"]).unsqueeze(0).to(device),
        "hand": torch.from_numpy(obs["hand"]).unsqueeze(0).to(device),
        "scalars": torch.from_numpy(obs["scalars"]).unsqueeze(0).to(device),
        "action_mask": torch.from_numpy(obs["action_mask"]).unsqueeze(0).to(device),
    }


def stack_obs(obs_list, device="cpu"):
    """List of observation dicts -> dict of (B, ...) tensors, for batched updates."""
    def cat(key, dtype=torch.float32):
        arr = np.stack([o[key] for o in obs_list])
        return torch.as_tensor(arr, dtype=dtype, device=device)

    return {
        "units_self": cat("units_self"),
        "units_enemy": cat("units_enemy"),
        "hand": cat("hand"),
        "scalars": cat("scalars"),
        "action_mask": cat("action_mask", dtype=torch.bool),
    }
