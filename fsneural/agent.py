"""Agent wrapper: turns the network into something that acts in the env.

Handles per-player recurrent memory and action sampling. In self-play one
PolicyAgent instance plays *both* sides, but each side keeps its own GRU memory
(they observe different things), so we key memory by player id.
"""

import torch
import torch.nn.functional as F

from .encoding import obs_to_tensors


class PolicyAgent:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.memory = {}  # player id -> (1, mem_size) tensor

    def reset_memory(self, players=(0, 1)):
        for p in players:
            self.memory[p] = self.model.initial_memory(1, self.device)

    @torch.no_grad()
    def act(self, obs, player, deterministic=False):
        """Pick an action for `player`. Returns (action, logp, value)."""
        self.model.eval()
        tensors = obs_to_tensors(obs, self.device)
        mem = self.memory.get(player)
        if mem is None:
            mem = self.model.initial_memory(1, self.device)
        logits, value, mem = self.model(tensors, mem)
        self.memory[player] = mem

        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = int(torch.argmax(probs, dim=-1).item())
        else:
            action = int(torch.multinomial(probs, 1).item())
        logp = torch.log(probs[0, action] + 1e-12).item()
        return action, logp, float(value.item())
