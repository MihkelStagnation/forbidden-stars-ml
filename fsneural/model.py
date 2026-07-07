"""Policy/value network for the combat env.

Architecture (the stack we discussed, shrunk to the combat mini-game):

    per-entity embeddings  (units + cards are variable-size *sets*)
            │
    self-attention encoder (interactions between units / cards)
            │
    recurrent memory (GRU) (belief state across a hidden-info episode)
            │
    ┌───────┴────────┐
  policy head      value head

The policy is a POINTER head: each hand-card slot's logit is computed from
that card's own encoded token (conditioned on the GRU memory), and likewise
each damage-assign logit from that unit's token. This matters because decks
are shuffled — a fixed slot->logit mapping from the pooled summary cannot
know which card sits in which slot; pointing at entity tokens can. (A single
monolithic head trained to exactly random-baseline strength; the pointer head
is the fix. Full-game compound actions will need autoregressive pointer
heads — same building block.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .combat_env import (
    UNIT_FEATS, CARD_FEATS, SCALAR_FEATS, ACTION_DIM, MAX_UNITS, MAX_HAND,
)

NEG_INF = -1e9


class CombatNet(nn.Module):
    """Policy/value net for the combat env, and (with `scalar_feats` and
    `n_build_actions` overridden) for the campaign env. Build actions have
    STABLE index meanings (buy tier t / buy card id c / pass), so they get a
    plain head off the memory — the pointer treatment is only needed where
    slot identity shuffles (hand cards, unit targets)."""

    def __init__(self, d_model=64, n_heads=4, mem_size=64,
                 scalar_feats=SCALAR_FEATS, n_build_actions=0):
        super().__init__()
        self.d_model = d_model
        self.mem_size = mem_size
        self.n_build_actions = n_build_actions

        # Per-entity embeddings. Units carry a self/enemy flag already, so a
        # single unit embedder handles both sides.
        self.unit_embed = nn.Linear(UNIT_FEATS, d_model)
        self.card_embed = nn.Linear(CARD_FEATS, d_model)
        self.scalar_embed = nn.Sequential(
            nn.Linear(scalar_feats, d_model), nn.ReLU(),
        )

        # Self-attention over the full set of entities (units + cards + a
        # scalar/context token).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Recurrent memory across decision steps within an episode.
        self.gru = nn.GRUCell(d_model, mem_size)

        # Pointer heads: per-entity logits from [entity token, memory].
        self.hand_ptr = nn.Sequential(
            nn.Linear(d_model + mem_size, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.unit_ptr = nn.Sequential(
            nn.Linear(d_model + mem_size, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.build_head = (
            nn.Sequential(nn.Linear(mem_size, mem_size), nn.ReLU(),
                          nn.Linear(mem_size, n_build_actions))
            if n_build_actions else None
        )
        self.value_head = nn.Sequential(
            nn.Linear(mem_size, mem_size), nn.ReLU(),
            nn.Linear(mem_size, 1), nn.Tanh(),
        )

    def initial_memory(self, batch_size=1, device="cpu"):
        return torch.zeros(batch_size, self.mem_size, device=device)

    def _encode(self, obs):
        """Run embeddings + attention; return encoded tokens (B, T, d).

        Token layout: [scalar, units_self (MAX_UNITS), units_enemy (MAX_UNITS),
        hand (MAX_HAND)].
        """
        us = self.unit_embed(obs["units_self"])    # (B, MAX_UNITS, d)
        ue = self.unit_embed(obs["units_enemy"])   # (B, MAX_UNITS, d)
        cd = self.card_embed(obs["hand"])          # (B, MAX_HAND, d)
        sc = self.scalar_embed(obs["scalars"]).unsqueeze(1)  # (B, 1, d)

        tokens = torch.cat([sc, us, ue, cd], dim=1)  # (B, 1+2U+H, d)
        return self.encoder(tokens)                   # (B, T, d)

    def forward(self, obs, memory):
        """obs: dict of batched tensors. memory: (B, mem_size).

        Returns (policy_logits, value, new_memory).
        """
        encoded = self._encode(obs)                   # (B, T, d)
        summary = encoded.mean(dim=1)                 # (B, d_model)
        memory = self.gru(summary, memory)            # (B, mem_size)

        hand_tok = encoded[:, 1 + 2 * MAX_UNITS:]     # (B, MAX_HAND, d)
        self_tok = encoded[:, 1:1 + MAX_UNITS]        # (B, MAX_UNITS, d)
        mem_exp_h = memory.unsqueeze(1).expand(-1, MAX_HAND, -1)
        mem_exp_u = memory.unsqueeze(1).expand(-1, MAX_UNITS, -1)
        hand_logits = self.hand_ptr(
            torch.cat([hand_tok, mem_exp_h], dim=-1)).squeeze(-1)  # (B, H)
        unit_logits = self.unit_ptr(
            torch.cat([self_tok, mem_exp_u], dim=-1)).squeeze(-1)  # (B, U)

        parts = [hand_logits, unit_logits]
        if self.build_head is not None:
            parts.append(self.build_head(memory))       # (B, n_build)
        logits = torch.cat(parts, dim=1)                 # (B, total actions)
        mask = obs["action_mask"]
        logits = logits.masked_fill(~mask, NEG_INF)   # forbid illegal actions

        value = self.value_head(memory).squeeze(-1)   # (B,)
        return logits, value, memory
