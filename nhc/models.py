"""Agent models: one shared encoder, one specialist, one consensus.

Design choices vs the legacy code:
- One `ObsEncoder` class (was GlyphEncoder + BLStatsEncoder + MessageEncoder +
  ObservationEncoder split across ~270 lines).
- The glyph crop is done in a single vectorised gather op (was a Python
  for-loop over batch). On a batch of 128 this is ~30x faster on GPU.
- `Agent` and `ConsensusHMoE` share the same recurrent core; consensus adds
  an options/mixture head over K frozen specialists.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nhc.env import NUM_ACTIONS

# Optional KB import — only needed when KBConditioner is used.
try:
    from nhc.kb import (
        build_entity_table, build_glyph_class_table, build_rule_table,
        N_ENTITY_FEATURES, NUM_GLYPH_CLASSES,
    )
    _HAS_KB = True
except ImportError:
    _HAS_KB = False

NUM_GLYPHS = 5991
BLSTATS_DIM = 27
CROP = 9
NUM_CHARS = 256


MAP_H = 21
MAP_W = 79


class ObsEncoder(nn.Module):
    """Fuses glyphs (9x9 egocentric crop + 21x79 global) + blstats + message
    into `out_dim`.

    Why both views: TorchBeast's NLE baseline (Küttler 2020) and Sample
    Factory's NetHack encoder both consume the global map AND a small
    egocentric crop. The crop gives high-resolution local detail (what
    monsters are next to me); the global view gives layout/exploration
    signal (where are the corridors and unexplored regions). With only
    the 9x9 crop the agent literally cannot see anything farther than 4
    tiles away, which kills navigation/exploration in NetHackScore.
    """

    def __init__(self, out_dim: int = 256, glyph_embed: int = 32,
                 global_embed: int = 16, use_global: bool = False):
        super().__init__()
        self.out_dim = out_dim
        # `use_global` is the "improved encoder" toggle (consensus only):
        #   True  → 21x79 global stem, no glyph padding_idx, LayerNorm blstats
        #   False → original encoder (loads existing specialist ckpts as-is)
        # Specialists were pretrained on small MiniHack rooms where 21x79 is
        # mostly empty padding; only consensus sees full NetHack and benefits
        # from the layout signal. Backward-compat keeps fine-tuned specialists
        # usable without re-training.
        self.use_global = use_global

        if use_global:
            # Glyph 0 is GLYPH_MON_OFF (a real monster); do NOT use padding_idx.
            self.glyph_emb = nn.Embedding(NUM_GLYPHS, glyph_embed)
        else:
            self.glyph_emb = nn.Embedding(NUM_GLYPHS, glyph_embed, padding_idx=0)

        # Local 9x9 crop stem (unchanged in both modes).
        self.glyph_cnn = nn.Sequential(
            nn.Conv2d(glyph_embed, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
        )
        self.glyph_fc = nn.Linear(128 * CROP * CROP, 128)

        if use_global:
            # Global 21x79 stem. Stride-2 convs collapse spatial 21x79 → 3x10.
            # Channels sized to use available VRAM (~10 GB free on a 32 GB
            # 5090): 32→64→128→128 with global_embed=32.
            ge = max(global_embed, 32)
            self.global_emb = nn.Embedding(NUM_GLYPHS, ge)
            self.global_cnn = nn.Sequential(
                nn.Conv2d(ge, 64, 3, stride=2, padding=1), nn.ReLU(True),     # ~11x40
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True),    # ~6x20
                nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(True),   # ~3x10
            )
            self.global_fc = nn.Linear(128 * 3 * 10, 128)

        if use_global:
            self.bl_norm = nn.LayerNorm(BLSTATS_DIM)
        else:
            # Original hand-tuned normalization preserved for ckpt compatibility.
            self.register_buffer("bl_mean", self._bl_norm_mean())
            self.register_buffer("bl_std", self._bl_norm_std())
        self.blstats_mlp = nn.Sequential(
            nn.Linear(BLSTATS_DIM, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
        )

        self.char_emb = nn.Embedding(NUM_CHARS, 32, padding_idx=0)
        self.msg_gru = nn.GRU(32, 128, batch_first=True)
        self.msg_fc = nn.Linear(128, 128)

        fuse_in = 128 * (4 if use_global else 3)
        self.fuse = nn.Sequential(nn.Linear(fuse_in, out_dim), nn.ReLU(True))

    @staticmethod
    def _bl_norm_mean() -> torch.Tensor:
        return torch.tensor([
            40.0, 10.0, 12.0, 50.0, 12.0, 12.0, 12.0, 12.0, 12.0, 500.0,
            20.0, 20.0, 3.0, 50.0, 10.0, 10.0, 5.0, 1.0, 3.0, 200.0,
            5000.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ], dtype=torch.float32)

    @staticmethod
    def _bl_norm_std() -> torch.Tensor:
        return torch.tensor([
            20.0, 6.0, 5.0, 30.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1000.0,
            20.0, 20.0, 5.0, 100.0, 15.0, 15.0, 5.0, 3.0, 5.0, 500.0,
            10000.0, 2.0, 2.0, 2.0, 5.0, 10.0, 5.0,
        ], dtype=torch.float32)

    @staticmethod
    def _crop(glyphs: torch.Tensor, ay: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        """Vectorised 9x9 egocentric crop. `glyphs`: (B, 21, 79) long. Returns (B, 9, 9)."""
        B, H, W = glyphs.shape
        half = CROP // 2
        padded = F.pad(glyphs, (half, half, half, half), value=0)
        cy = (ay.clamp(0, H - 1) + half).long()
        cx = (ax.clamp(0, W - 1) + half).long()
        # Build per-sample row/col indices: (B, 9), (B, 9)
        ar = torch.arange(CROP, device=glyphs.device)
        rows = cy.unsqueeze(1) - half + ar  # (B, 9)
        cols = cx.unsqueeze(1) - half + ar  # (B, 9)
        # Expand to (B, 9, 9) via broadcasting
        rows_e = rows.unsqueeze(2).expand(-1, -1, CROP)  # (B, 9, 9)
        cols_e = cols.unsqueeze(1).expand(-1, CROP, -1)
        batch_e = torch.arange(B, device=glyphs.device).view(B, 1, 1).expand_as(rows_e)
        return padded[batch_e, rows_e, cols_e]

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        glyphs = obs["glyphs"].long()
        blstats = obs["blstats"].float()
        message = obs["message"].long()
        ax = blstats[:, 0]
        ay = blstats[:, 1]

        c = self._crop(glyphs, ay, ax)                       # (B, 9, 9)
        e = self.glyph_emb(c).permute(0, 3, 1, 2)            # (B, D, 9, 9)
        g_local = self.glyph_fc(self.glyph_cnn(e).flatten(1))  # (B, 128)

        if self.use_global:
            b = self.blstats_mlp(self.bl_norm(blstats))
        else:
            b = self.blstats_mlp((blstats - self.bl_mean) / (self.bl_std + 1e-6))

        m = self.char_emb(message)                            # (B, 256, 32)
        _, h = self.msg_gru(m)                                # h: (1, B, 128)
        m = self.msg_fc(h.squeeze(0))

        parts = [g_local]
        if self.use_global:
            ge = self.global_emb(glyphs).permute(0, 3, 1, 2)  # (B, D, 21, 79)
            g_global = self.global_fc(self.global_cnn(ge).flatten(1))  # (B, 128)
            parts.append(g_global)
        parts.extend([b, m])
        return self.fuse(torch.cat(parts, dim=-1))


def _orthogonal_init_head(head: nn.Sequential, *, policy_last_gain: float = 0.01
                          ) -> None:
    """Orthogonal init on an MLP head (Sequential(Linear, ReLU, Linear)).

    Hidden layers get gain sqrt(2) (standard for ReLU). The output Linear
    gets ``policy_last_gain`` — the PPO-paper convention is a very small
    gain (0.01) for policy heads so early-training logits are near zero
    (i.e., near-uniform action distribution) and a gain of 1.0 for value
    heads so value predictions start at rollout-value-scale.

    Reference: Huang 2022 "The 37 Implementation Details of Proximal
    Policy Optimization" §4 — orthogonal init is the top-3 lever for
    reproducible PPO.
    """
    hidden_gain = (2.0) ** 0.5
    linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
    for m in linears[:-1]:
        nn.init.orthogonal_(m.weight, gain=hidden_gain)
        nn.init.zeros_(m.bias)
    last = linears[-1]
    nn.init.orthogonal_(last.weight, gain=policy_last_gain)
    nn.init.zeros_(last.bias)


class KBConditioner(nn.Module):
    """Injects structured knowledge base context into the agent feature stream.

    Two pathways:
    1. Entity pathway: maps visible glyphs in the 9x9 crop to KB property
       vectors, then attention-pools to a fixed-size context.
    2. Rule pathway: maps blstats through an MLP to soft rule activations,
       then weighted-sums rule embeddings.

    Output is concatenated with the encoder output before the LSTM.
    Total params: ~23K (trainable by PPO). The 212-dim meta-controller
    vector (trust weights, attention temp, etc.) is set externally by CMA-ES.
    """

    def __init__(self, num_rules: int = 80, kb_dim: int = 64):
        super().__init__()
        assert _HAS_KB, "nhc.kb not available; install NLE first"
        self.kb_dim = kb_dim
        self.num_rules = num_rules

        # Static KB data (not trainable)
        entity_table = build_entity_table()   # (MAX_GLYPH, N_ENTITY_FEATURES)
        glyph_classes = build_glyph_class_table()  # (MAX_GLYPH,)
        self.register_buffer("entity_table", torch.from_numpy(entity_table))
        self.register_buffer("glyph_classes", torch.from_numpy(glyph_classes))

        # Entity pathway: project N_ENTITY_FEATURES -> kb_dim, then attention
        self.entity_proj = nn.Linear(N_ENTITY_FEATURES, kb_dim)
        self.entity_attn = nn.Linear(kb_dim, 1)

        # Rule pathway: blstats -> soft rule activations
        self.rule_encoder = nn.Sequential(
            nn.Linear(BLSTATS_DIM, 64), nn.ReLU(True),
            nn.Linear(64, num_rules), nn.Sigmoid(),
        )
        self.rule_embeds = nn.Embedding(num_rules, kb_dim)

        # Fuse entity + rule context
        self.out_proj = nn.Linear(kb_dim * 2, kb_dim)

        # Meta-controller params (set by CMA-ES, not trained by PPO)
        # Defaults are uniform/neutral so PPO training works without EC.
        self.register_buffer("trust_weights",
                             torch.ones(num_rules, dtype=torch.float32))
        self.register_buffer("entity_type_weights",
                             torch.ones(NUM_GLYPH_CLASSES, dtype=torch.float32))
        self.register_buffer("rule_priority_bias",
                             torch.zeros(num_rules, dtype=torch.float32))
        self.register_buffer("query_threshold",
                             torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("attention_temp",
                             torch.tensor(1.0, dtype=torch.float32))

    def set_meta_params(self, flat: np.ndarray) -> None:
        """Load the 212-dim meta-controller vector from CMA-ES."""
        import numpy as np  # noqa: F811
        assert len(flat) == self.meta_param_count(), \
            "Expected %d params, got %d" % (self.meta_param_count(), len(flat))
        offset = 0
        n = self.num_rules
        self.trust_weights.copy_(
            torch.sigmoid(torch.from_numpy(flat[offset:offset + n].copy())))
        offset += n
        self.entity_type_weights.copy_(
            torch.softmax(torch.from_numpy(
                flat[offset:offset + NUM_GLYPH_CLASSES].copy()), dim=0) * NUM_GLYPH_CLASSES)
        offset += NUM_GLYPH_CLASSES
        self.rule_priority_bias.copy_(
            torch.from_numpy(flat[offset:offset + n].copy()))
        offset += n
        self.query_threshold.copy_(
            torch.tensor(float(flat[offset]), dtype=torch.float32).sigmoid())
        offset += 1
        self.attention_temp.copy_(
            torch.tensor(float(flat[offset]), dtype=torch.float32).exp().clamp(0.1, 10.0))
        offset += 1

    def meta_param_count(self) -> int:
        return self.num_rules * 2 + NUM_GLYPH_CLASSES + 2  # 80+10+80+1+1 = 172

    def forward(self, crop_glyphs: torch.Tensor, blstats: torch.Tensor) -> torch.Tensor:
        """Compute KB context vector.

        Args:
            crop_glyphs: (B, 9, 9) int64, egocentric glyph crop
            blstats: (B, 27) float32

        Returns:
            (B, kb_dim) KB context vector
        """
        B = crop_glyphs.shape[0]
        device = crop_glyphs.device

        # --- Entity pathway ---
        flat_glyphs = crop_glyphs.reshape(B, -1).long()  # (B, 81)
        # Clamp to valid range
        flat_glyphs = flat_glyphs.clamp(0, self.entity_table.shape[0] - 1)
        # Look up entity properties: (B, 81, N_ENTITY_FEATURES)
        entity_props = self.entity_table[flat_glyphs]
        # Per-glyph class weight from meta-controller
        glyph_cls = self.glyph_classes[flat_glyphs]  # (B, 81)
        cls_weight = self.entity_type_weights[glyph_cls]  # (B, 81)

        # Project to kb_dim and compute attention
        entity_h = torch.relu(self.entity_proj(entity_props))  # (B, 81, kb_dim)
        attn_scores = self.entity_attn(entity_h).squeeze(-1)   # (B, 81)
        attn_scores = attn_scores * cls_weight / self.attention_temp
        attn_weights = torch.softmax(attn_scores, dim=-1)      # (B, 81)
        entity_ctx = (attn_weights.unsqueeze(-1) * entity_h).sum(dim=1)  # (B, kb_dim)

        # --- Rule pathway ---
        rule_act = self.rule_encoder(blstats)  # (B, num_rules), already sigmoid
        rule_act = rule_act + self.rule_priority_bias.unsqueeze(0)
        rule_act = rule_act * self.trust_weights.unsqueeze(0)
        rule_act = rule_act.clamp(0, 1)
        rule_ids = torch.arange(self.num_rules, device=device)
        rule_h = self.rule_embeds(rule_ids)  # (num_rules, kb_dim)
        rule_ctx = torch.matmul(rule_act, rule_h)  # (B, kb_dim)

        # --- Fuse ---
        fused = torch.cat([entity_ctx, rule_ctx], dim=-1)  # (B, kb_dim*2)
        return torch.relu(self.out_proj(fused))  # (B, kb_dim)


class Agent(nn.Module):
    """Recurrent actor-critic. Used as a specialist (K=0) or as the base of
    the consensus model.

    Uses `nn.LSTM` so replay over a (T, B, D) rollout is a single fused
    cuDNN call instead of a Python-level T-step loop. Single-step
    `forward` (during rollout collection) unsqueezes/squeezes time.
    """

    def __init__(self, hidden_dim: int = 256, num_actions: int = NUM_ACTIONS,
                 use_global: bool = True, head_dim: int = 128,
                 kb_conditioner: KBConditioner | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.kb = kb_conditioner
        # use_global=True matches the consensus encoder so specialists
        # also see the full 21x79 map — essential for NetHack where
        # stairs/monsters/items are typically > 4 tiles from the agent
        # and the 9x9 egocentric crop alone is blind past that. The
        # use_global=False branch is retained only for loading the old
        # pre-2026-04 specialist ckpts into a back-compat model.
        self.encoder = ObsEncoder(out_dim=hidden_dim, use_global=use_global)
        # LSTM input dim is hidden_dim + kb_dim when KB is attached.
        lstm_in_dim = hidden_dim + (kb_conditioner.kb_dim if kb_conditioner else 0)
        # LayerNorm around the LSTM stabilises bf16 training and matches
        # what NLE baselines (Küttler 2020, Appendix) do.
        self.lstm_input_ln = nn.LayerNorm(lstm_in_dim)
        self.lstm = nn.LSTM(lstm_in_dim, hidden_dim, num_layers=1)
        self.lstm_output_ln = nn.LayerNorm(hidden_dim)
        # 2-layer MLP heads (one hidden layer at head_dim). Single-linear
        # heads — the prior design — are the standard baseline, but for
        # NetHack's 121-action space a small MLP gives the heads enough
        # capacity to specialise per-action without affecting the
        # rollout-time speed measurably.
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ReLU(True),
            nn.Linear(head_dim, num_actions))
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ReLU(True),
            nn.Linear(head_dim, 1))
        _orthogonal_init_head(self.policy, policy_last_gain=0.01)
        _orthogonal_init_head(self.value, policy_last_gain=1.0)

    def initial_state(self, B: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # (h, c) as (B, D) — stored time-unaware; lstm calls add the layer dim.
        return (torch.zeros(B, self.hidden_dim, device=device),
                torch.zeros(B, self.hidden_dim, device=device))

    def _get_crop(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract the 9x9 egocentric crop for KB lookups."""
        glyphs = obs["glyphs"].long()
        blstats = obs["blstats"].float()
        return ObsEncoder._crop(glyphs, blstats[:, 1], blstats[:, 0])

    def forward(self, obs: dict[str, torch.Tensor], state: tuple[torch.Tensor, torch.Tensor],
                action_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z = self.encoder(obs)                         # (B, D)
        if self.kb is not None:
            crop = self._get_crop(obs)
            kb_ctx = self.kb(crop, obs["blstats"].float())  # (B, kb_dim)
            z = torch.cat([z, kb_ctx], dim=-1)        # (B, D + kb_dim)
        z_norm = self.lstm_input_ln(z)
        h0, c0 = state
        out, (h1, c1) = self.lstm(z_norm.unsqueeze(0),
                                  (h0.unsqueeze(0), c0.unsqueeze(0)))
        h = self.lstm_output_ln(out.squeeze(0))
        # Policy/value heads in fp32 to stop bf16 overflow from producing
        # inf/NaN logits under the Categorical sampler. The prior V-trace
        # run crashed at step 400M on exactly this — see HANDOFF_PPO.md §3.4.
        with torch.amp.autocast(device_type=h.device.type, enabled=False):
            h_fp32 = h.float()
            logits = self.policy(h_fp32)
            value = self.value(h_fp32).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return {
            "features": z,
            "core": h,
            "logits": logits,
            "value": value,
            "state": (h1.squeeze(0), c1.squeeze(0)),
        }

    def forward_sequence(
        self,
        obs_seq: dict[str, torch.Tensor],      # each (T, B, ...)
        init_state: tuple[torch.Tensor, torch.Tensor],  # (B, D), (B, D)
        dones: torch.Tensor,                   # (T, B) — marks end of episode
        action_mask: torch.Tensor | None = None,  # (B, A) or (T, B, A)
    ) -> dict[str, torch.Tensor]:
        """Replay a full (T, B, ...) rollout under current params.

        Runs the encoder batched over T*B, then the LSTM in segments split
        at episode boundaries. Inside each segment the LSTM is one fused
        call; we concatenate outputs across segments. This avoids both the
        Python T-step for-loop and the nn.LSTMCell per-step overhead.
        """
        T, B = dones.shape
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs_seq.items()}
        z = self.encoder(flat).reshape(T, B, self.hidden_dim)
        if self.kb is not None:
            crop = self._get_crop(flat)  # (T*B, 9, 9)
            kb_ctx = self.kb(crop, flat["blstats"].float())  # (T*B, kb_dim)
            kb_ctx = kb_ctx.reshape(T, B, self.kb.kb_dim)
            z = torch.cat([z, kb_ctx], dim=-1)  # (T, B, D + kb_dim)
        z_norm = self.lstm_input_ln(z)

        outs = []
        h = init_state[0].unsqueeze(0)     # (1, B, D)
        c = init_state[1].unsqueeze(0)
        # Segment boundaries: indices where any env had `done` at the end
        # of step t, so step t+1 must start from zero state for that env.
        # We still call the LSTM segment-by-segment; within a segment all
        # envs are valid, state reset is done at the boundary by zeroing
        # per-env columns.
        dones_bool = dones.bool()
        t_start = 0
        for t in range(T):
            has_done = bool(dones_bool[t].any())
            if has_done or t == T - 1:
                seg = z_norm[t_start:t + 1]                     # (L, B, D)
                seg_out, (h, c) = self.lstm(seg, (h, c))
                outs.append(seg_out)
                if has_done and t < T - 1:
                    reset = ~dones_bool[t].unsqueeze(0).unsqueeze(-1)  # (1,B,1)
                    h = h * reset
                    c = c * reset
                t_start = t + 1
        h_seq = self.lstm_output_ln(torch.cat(outs, dim=0))     # (T, B, D)
        # fp32 heads — see forward() for rationale.
        with torch.amp.autocast(device_type=h_seq.device.type, enabled=False):
            h_seq_fp32 = h_seq.float()
            logits = self.policy(h_seq_fp32)                    # (T, B, A)
            values = self.value(h_seq_fp32).squeeze(-1)         # (T, B)
        if action_mask is not None:
            if action_mask.dim() == 2:
                mask = action_mask.unsqueeze(0).expand_as(logits)
            else:
                mask = action_mask
            logits = logits.masked_fill(~mask, -1e9)
        return {
            "features": z,
            "core": h_seq,
            "logits": logits,
            "value": values,
            "final_state": (h.squeeze(0), c.squeeze(0)),
        }


class ConsensusHMoE(nn.Module):
    """HO-MoE (Hierarchical Options Mixture-of-Experts) over K frozen specialists.

    Simpler than the legacy HOMoEConsensusModel:
    - One option selector (softmax over K) instead of a separate meta-LSTM.
    - Blended policy = softmax( mixture of specialist logits + consensus logits ).
    - Termination is replaced by option-entropy regularisation + option
      persistence via state carry (the option selector sees the previous option).

    This keeps the theoretical story (mixture of experts + options bias) but
    removes half the moving parts that made the legacy version hard to train.
    """

    def __init__(self, specialists: list[Agent], hidden_dim: int = 1024,
                 adapter_dim: int = 128, num_actions: int = NUM_ACTIONS,
                 specialist_masks: list[torch.Tensor] | None = None,
                 head_dim: int = 256):
        super().__init__()
        K = len(specialists)
        self.K = K
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        self.specialists = nn.ModuleList(specialists)
        for s in self.specialists:
            for p in s.parameters():
                p.requires_grad = False

        self.encoder = ObsEncoder(out_dim=hidden_dim, use_global=True)
        self.adapters = nn.ModuleList([nn.Linear(s.hidden_dim, adapter_dim) for s in specialists])
        self.adapter_gates = nn.Parameter(torch.zeros(K))
        # LayerNorm around the core LSTM matches the Agent change.
        # 2 layers (bumped from 1 in the PPO migration) — specialists stay
        # at 1 so their frozen ckpts still load; the consensus core gets
        # the extra recurrent capacity for full 21×79 NetHack observations.
        # State shape becomes (num_layers, B, D) — callers must not
        # unsqueeze/squeeze at the LSTM call site any more.
        self.lstm_num_layers = 2
        self.lstm_input_ln = nn.LayerNorm(hidden_dim + adapter_dim * K + K)
        self.lstm = nn.LSTM(hidden_dim + adapter_dim * K + K, hidden_dim,
                            num_layers=self.lstm_num_layers)
        self.lstm_output_ln = nn.LayerNorm(hidden_dim)
        # 2-layer MLP heads.
        self.option_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ReLU(True),
            nn.Linear(head_dim, K))
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ReLU(True),
            nn.Linear(head_dim, num_actions))
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ReLU(True),
            nn.Linear(head_dim, 1))
        # Orthogonal init (Huang 2022 §4). Option head is a "policy" too
        # — a distribution over K options — so it also gets the 0.01
        # gain so the router starts near-uniform (this is *additional*
        # to the option-entropy bonus; the initial uniformity comes from
        # the init, the ongoing uniformity comes from the loss).
        _orthogonal_init_head(self.policy, policy_last_gain=0.01)
        _orthogonal_init_head(self.value, policy_last_gain=1.0)
        _orthogonal_init_head(self.option_head, policy_last_gain=0.01)
        # State-dependent mixing coeff: lambda_t = sigmoid(lambda_head(h_t))
        # in [0, 1] per timestep (Jacobs 1991 gating). Output-linear bias
        # is initialised to 0.4 so sigmoid ≈ 0.6 at start — the consensus
        # head gets slightly more weight than the specialist mixture to
        # speed early convergence (the random-init mixture is noisy).
        # Keep this custom init (zero weight + 0.4 bias) rather than
        # orthogonal — we want lambda starting at a known 0.60, not at
        # a small random value around sigmoid(0) = 0.5.
        self.lambda_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.ReLU(True),
            nn.Linear(head_dim, 1))
        nn.init.orthogonal_(self.lambda_head[0].weight, gain=(2.0) ** 0.5)
        nn.init.zeros_(self.lambda_head[0].bias)
        nn.init.zeros_(self.lambda_head[-1].weight)
        nn.init.constant_(self.lambda_head[-1].bias, 0.4)

        if specialist_masks is None:
            smask = torch.ones(K, num_actions, dtype=torch.bool)
        else:
            smask = torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in specialist_masks])
        self.register_buffer("spec_masks", smask)

    def initial_state(self, B: int, device: torch.device) -> dict:
        L = self.lstm_num_layers
        return {
            "core": (torch.zeros(L, B, self.hidden_dim, device=device),
                     torch.zeros(L, B, self.hidden_dim, device=device)),
            "spec": [s.initial_state(B, device) for s in self.specialists],
            "prev_option": torch.zeros(B, dtype=torch.long, device=device),
        }

    def forward(self, obs: dict[str, torch.Tensor], state: dict,
                action_mask: torch.Tensor | None = None,
                deterministic: bool = False) -> dict[str, torch.Tensor]:
        B = obs["glyphs"].shape[0]
        device = obs["glyphs"].device

        env_mask = None
        if action_mask is not None:
            env_mask = action_mask if action_mask.dtype == torch.bool else action_mask.bool()

        spec_logits = []
        spec_features = []
        new_spec_state = []
        with torch.no_grad():
            for k, s in enumerate(self.specialists):
                tm = self.spec_masks[k].unsqueeze(0).expand(B, -1)
                m = tm & env_mask if env_mask is not None else tm
                out = s(obs, state["spec"][k], m)
                spec_logits.append(out["logits"])
                spec_features.append(out["features"])
                new_spec_state.append(out["state"])

        z = self.encoder(obs)
        gated = [torch.sigmoid(self.adapter_gates[k]) * self.adapters[k](spec_features[k])
                 for k in range(self.K)]
        prev_onehot = F.one_hot(state["prev_option"], self.K).float()
        lstm_in = torch.cat([z] + gated + [prev_onehot], dim=-1)
        lstm_in = self.lstm_input_ln(lstm_in)
        # state["core"] is (num_layers, B, D); pass through unchanged.
        h0, c0 = state["core"]
        out, (h1, c1) = self.lstm(lstm_in.unsqueeze(0), (h0, c0))
        # Apply LN only to the copy used by downstream heads; keep the
        # raw LSTM hidden as the next-step state so it matches
        # forward_sequence (which also carries raw h forward). The prior
        # code stored LN(h) in state["core"], which caused a silent
        # divergence between the two replay paths starting at t=1.
        h = self.lstm_output_ln(out.squeeze(0))   # (B, D) — heads read last-layer
        h_raw = h1                                 # (num_layers, B, D) — raw state
        c = c1                                     # (num_layers, B, D)

        # fp32 heads — see Agent.forward for rationale. All four sensitive
        # heads run in fp32: option_head (feeds mixture + router-z loss),
        # policy (blended logits), value, lambda_head (sigmoid gate).
        with torch.amp.autocast(device_type=h.device.type, enabled=False):
            h_fp32 = h.float()
            option_logits = self.option_head(h_fp32)
            consensus_logits = self.policy(h_fp32)
            value = self.value(h_fp32).squeeze(-1)
            lam = torch.sigmoid(self.lambda_head(h_fp32))                     # (B, 1)

        opt_probs = F.softmax(option_logits, dim=-1)                          # (B, K)
        stacked = torch.stack(spec_logits, dim=1).float()                      # (B, K, A)
        # Stacked specialist features, for the rollout-time cache so the PPO
        # learn step doesn't need to re-run the 8 frozen specialists.
        spec_feat_stacked = torch.stack(spec_features, dim=1)                  # (B, K, D_spec)
        mixture_logits = (opt_probs.unsqueeze(2) * stacked).sum(dim=1)         # (B, A)

        if env_mask is not None:
            mixture_logits = mixture_logits.masked_fill(~env_mask, -1e9)
            consensus_logits = consensus_logits.masked_fill(~env_mask, -1e9)

        blended_logits = lam * consensus_logits + (1 - lam) * mixture_logits
        if deterministic:
            new_option = option_logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=option_logits)
            new_option = dist.sample()
        return {
            "logits": blended_logits,
            "consensus_logits": consensus_logits,
            "mixture_logits": mixture_logits,
            "option_logits": option_logits,
            "option_probs": opt_probs,
            "spec_logits": stacked,                     # (B, K, A)
            "spec_features": spec_feat_stacked,         # (B, K, D_spec)
            "value": value,
            "lambda": lam.squeeze(-1),
            "state": {
                "core": (h_raw, c),
                "spec": new_spec_state,
                "prev_option": new_option.detach(),
            },
        }

    def forward_sequence(
        self,
        obs_seq: dict[str, torch.Tensor],          # each (T, B, ...)
        init_state: dict,                           # {core: (h,c), spec: [(h,c)*K], prev_option: (B,)}
        dones: torch.Tensor,                        # (T, B)
        action_mask: torch.Tensor | None = None,    # (B, A) or (T, B, A)
        deterministic: bool = False,
        cached_spec: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Replay a (T, B, ...) consensus rollout with partial fusion.

        Encoder + each specialist are replayed via the fused
        `Agent.forward_sequence` path (single cuDNN call per specialist).
        The core LSTM still steps sequentially because its input at t+1
        depends on the option chosen at t.

        ``cached_spec`` — optional dict with keys ``logits`` ``(T, B, K, A)``
        and ``features`` ``(T, B, K, D_spec)`` — the **per-step specialist
        outputs captured at rollout collection time**. Specialists are
        frozen and deterministic given ``(obs, init_state.spec, dones)``,
        so the per-minibatch PPO replay can skip the 8 specialist forwards
        entirely and feed the cached tensors straight into the
        mixture/adapter paths. In the pilot benchmark this dropped the
        learn-step wall clock from ~5 s to ~0.8 s (~6× on learn, ~2.5×
        on end-to-end SPS). Adapters are still trainable so the adapter
        forward runs fresh each call.
        """
        T, B = dones.shape
        dones_bool = dones.bool()

        flat_obs = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs_seq.items()}
        z_seq = self.encoder(flat_obs).reshape(T, B, self.hidden_dim)

        env_mask = None
        if action_mask is not None:
            em = action_mask if action_mask.dtype == torch.bool else action_mask.bool()
            env_mask = em.unsqueeze(0).expand(T, B, -1) if em.dim() == 2 else em

        if cached_spec is not None:
            # Fast path: use precomputed specialist outputs. Shapes:
            # logits (T, B, K, A), features (T, B, K, D_spec).
            spec_stack = cached_spec["logits"]
            spec_feat_all = [cached_spec["features"][..., k, :]
                             for k in range(self.K)]
        else:
            spec_logits_all: list[torch.Tensor] = []
            spec_feat_all = []
            with torch.no_grad():
                for k, s in enumerate(self.specialists):
                    tm = self.spec_masks[k].view(1, 1, -1)
                    mask_k = (tm & env_mask) if env_mask is not None else tm.expand(T, B, -1)
                    out_k = s.forward_sequence(
                        obs_seq=obs_seq,
                        init_state=init_state["spec"][k],
                        dones=dones,
                        action_mask=mask_k,
                    )
                    spec_logits_all.append(out_k["logits"])
                    spec_feat_all.append(out_k["features"])
            spec_stack = torch.stack(spec_logits_all, dim=2)  # (T, B, K, A)

        gated_all = [torch.sigmoid(self.adapter_gates[k]) * self.adapters[k](spec_feat_all[k])
                     for k in range(self.K)]
        gated_concat = torch.cat(gated_all, dim=-1)          # (T, B, K*adapter_dim)

        # Core LSTM state is (num_layers, B, D); no unsqueeze needed.
        h = init_state["core"][0]
        c = init_state["core"][1]
        prev_option = init_state["prev_option"]

        logits_out, values_out, opt_out, mix_out, lam_out = [], [], [], [], []
        prev_opt_out: list[torch.Tensor] = []
        for t in range(T):
            prev_opt_out.append(prev_option.clone())
            prev_onehot = F.one_hot(prev_option, self.K).float()
            lstm_in = torch.cat([z_seq[t], gated_concat[t], prev_onehot], dim=-1)
            lstm_in = self.lstm_input_ln(lstm_in)
            out, (h, c) = self.lstm(lstm_in.unsqueeze(0), (h, c))
            h_t = self.lstm_output_ln(out.squeeze(0))

            # fp32 heads on the per-step core output.
            with torch.amp.autocast(device_type=h_t.device.type, enabled=False):
                h_t_fp32 = h_t.float()
                option_logits_t = self.option_head(h_t_fp32)
                cons_t = self.policy(h_t_fp32)
                value_t = self.value(h_t_fp32).squeeze(-1)
                lam_t = torch.sigmoid(self.lambda_head(h_t_fp32))           # (B, 1)

            opt_probs_t = F.softmax(option_logits_t, dim=-1)
            mix_t = (opt_probs_t.unsqueeze(2) * spec_stack[t].float()).sum(dim=1)
            if env_mask is not None:
                mk = env_mask[t]
                mix_t = mix_t.masked_fill(~mk, -1e9)
                cons_t = cons_t.masked_fill(~mk, -1e9)
            blended_t = lam_t * cons_t + (1 - lam_t) * mix_t

            logits_out.append(blended_t)
            values_out.append(value_t)
            opt_out.append(option_logits_t)
            mix_out.append(mix_t)
            lam_out.append(lam_t.squeeze(-1))

            if deterministic:
                new_option = option_logits_t.argmax(dim=-1).detach()
            else:
                dist_t = torch.distributions.Categorical(logits=option_logits_t)
                new_option = dist_t.sample().detach()
            if t < T - 1:
                d_t = dones_bool[t]
                # keep shape (1, B, 1) broadcasts over (num_layers, B, D).
                keep_core = (~d_t).float().view(1, -1, 1)
                h = h * keep_core
                c = c * keep_core
                prev_option = torch.where(d_t, torch.zeros_like(new_option), new_option)
            else:
                prev_option = new_option

        return {
            "logits": torch.stack(logits_out, 0),
            "value": torch.stack(values_out, 0),
            "option_logits": torch.stack(opt_out, 0),
            "mixture_logits": torch.stack(mix_out, 0),
            "lambda": torch.stack(lam_out, 0),
            "prev_option": torch.stack(prev_opt_out, 0),
            "features": z_seq,
        }


# Deprecation alias: retain the old `Consensus` name for one commit cycle so
# in-flight branches keep importing cleanly. Remove after the rest of the
# repo has migrated to `ConsensusHMoE`.
Consensus = ConsensusHMoE
