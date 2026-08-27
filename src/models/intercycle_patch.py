"""Inter-Cycle Patch Transformer (ICPT) residual head and simple cycle baselines.

The inter-cycle experiment plan replaces the NLinear residual branch of the
frozen RCRF candidate with a transformer that treats each complete cycle as a
token and models the evolution of full cycle blocks, then compares position
encodings on that same model.

Overview of the module:

- ``RepeatLastCycleResidualHead`` (A3): repeats the most recent complete cycle.
- ``CycleNetStyleResidualHead`` (A4): learnable cycle template + residual NLinear.
- ``InterCyclePatchResidualHead`` (A5 + P0-P9): a compact encoder/decoder
  transformer over cycle tokens, ``W_out`` zero-initialized so the initial
  output is exactly RepeatLastCycle.  The position encoding is selected by
  ``pe_type``:

    - token-level: ``none``, ``sincos``, ``learned_abs``, ``time2vec``,
      ``calendar`` (marks required)
    - attention-level: ``rope``, ``relative``, ``alibi``, ``lff``
    - combined: ``sincos_relative`` (P1 token + P5 pairwise bias)

The head is channel-independent with shared weights: ``B`` and ``C`` are merged
into ``B*C`` and all channels use the same encoder/decoder, so Electricity /
Traffic do not scale parameters with the channel count.

Design constraints carried over from the plan:

- ``W_out=0`` -> output equals RepeatLastCycle (warm start).
- No future values are consumed: future cycles are predicted from learned
  query tokens plus cross-attention over history tokens only.
- Calendar PE reads only the timestamp marks already provided to the model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ICPTPE_TYPES = (
    "none",
    "sincos",
    "learned_abs",
    "time2vec",
    "rope",
    "relative",
    "alibi",
    "lff",
    "sincos_relative",
    "calendar",
)

# Which PE component each pe_type activates.
_TOKEN_PE = {
    "none": "none",
    "sincos": "sincos",
    "learned_abs": "learned_abs",
    "time2vec": "time2vec",
    "rope": "none",
    "relative": "none",
    "alibi": "none",
    "lff": "none",
    "sincos_relative": "sincos",
    "calendar": "calendar",
}
_ATTN_PE = {
    "none": "none",
    "sincos": "none",
    "learned_abs": "none",
    "time2vec": "none",
    "rope": "rope",
    "relative": "relative",
    "alibi": "alibi",
    "lff": "lff",
    "sincos_relative": "relative",
    "calendar": "none",
}


class RepeatLastCycleResidualHead(nn.Module):
    """A3: repeat the most recent complete cycle over the forecast horizon.

    Pure persistence baseline used to test whether the ICPT transformer's gain
    is more than "just repeat the last cycle".  No parameters.
    """

    def __init__(self, seq_len: int, pred_len: int, period_len: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)

    def forward(self, x):  # x: (B, L, C) normalized scale
        last_cycle = x[:, -self.period_len:, :]  # (B, P, C)
        repeats = (self.pred_len + self.period_len - 1) // self.period_len
        return last_cycle.repeat(1, repeats, 1)[:, : self.pred_len, :]


class CycleNetStyleResidualHead(nn.Module):
    """A4: CycleNet-style reference (learnable template + residual linear).

    Learns a per-phase cycle template ``t`` and extrapolates the cycle-removed
    residual history with an NLinear map.  At init ``t=0`` and the linear map
    is zero, so the output is exactly the NLinear warm start (last value
    repeated), making A4 comparable to A2/A3/A5 from the same starting point.
    """

    def __init__(self, seq_len: int, pred_len: int, period_len: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.template = nn.Parameter(torch.zeros(period_len))
        self.linear = nn.Linear(seq_len, pred_len)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def _template_lookup(self, length, device, dtype):
        idx = torch.arange(length, device=device) % self.period_len
        return self.template[idx].view(1, length, 1)  # (1, L, 1) broadcasts over C

    def forward(self, x):  # x: (B, L, C)
        cycle_hist = self._template_lookup(self.seq_len, x.device, x.dtype)
        resid_hist = x - cycle_hist
        last = x[:, -1:, :]
        centered = (resid_hist - resid_hist[:, -1:, :]).permute(0, 2, 1)
        delta = self.linear(centered).permute(0, 2, 1)  # (B, pred_len, C)
        cycle_fut = self._template_lookup(self.pred_len, x.device, x.dtype)
        return cycle_fut + delta + last.expand(-1, delta.size(1), -1)


class _ICPTAttention(nn.Module):
    """Multi-head attention with pluggable position bias.

    ``attn_pe`` selects the attention-level position mechanism: ``rope`` rotates
    Q/K by absolute cycle index, ``relative`` adds a bucketed pairwise bias,
    ``alibi`` adds a deterministic head-wise linear distance decay, and ``lff``
    adds a learnable Fourier-similarity bias over the pairwise cycle distance.
    All biases are computed from the query/key positions passed in.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_pe: str,
        relative_buckets: int,
        lff_frequencies: int,
        max_position: int,
        period_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_pe = attn_pe
        self.period_len = int(period_len)
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        if attn_pe == "relative":
            self.relative_buckets = int(relative_buckets)
            self.relative_bias = nn.Parameter(
                torch.zeros(self.relative_buckets, num_heads)
            )
        elif attn_pe == "alibi":
            slopes = 2.0 ** (-8.0 * torch.arange(1, num_heads + 1, dtype=torch.float32) / num_heads)
            self.register_buffer("alibi_slopes", slopes)
        elif attn_pe == "lff":
            num_freq = max(1, int(lff_frequencies))
            self.num_lff_freq = num_freq
            harmonic = torch.arange(1, num_freq + 1, dtype=torch.float32)
            base = 2.0 * math.pi * harmonic / max(float(max_position), 1.0)
            self.register_buffer("lff_base_frequency", base)
            self.lff_log_scale = nn.Parameter(torch.zeros(num_freq))
            # Zero init: the LFF bias starts neutral (warm start for P7).
            self.lff_proj = nn.Parameter(torch.zeros(num_heads, 2 * num_freq))
        elif attn_pe == "rope":
            half = self.head_dim // 2
            inv_freq = 1.0 / (
                10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half)
            )
            self.register_buffer("rope_inv_freq", inv_freq)

    def _rope(self, x, positions):
        # x: (N, H, S, d_h), positions: (S,)
        half = self.head_dim // 2
        angles = positions.float().unsqueeze(-1) * self.rope_inv_freq.unsqueeze(0)
        cos = angles.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, S, half)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)
        x1 = x[..., :half]
        x2 = x[..., half:]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)

    def _relative_bucket(self, d):
        # d: (Q, K) integer distance query_pos - key_pos
        half = self.relative_buckets // 2
        pos = d.abs().float()
        log_bucket = torch.where(
            pos < half, pos, pos.clamp_min(1.0).log2().floor().clamp(max=half - 1.0)
        )
        bucket = torch.where(d > 0, half + log_bucket, half - 1.0 - log_bucket)
        return bucket.clamp(0, self.relative_buckets - 1).long()

    def _bias(self, q_pos, kv_pos):
        if self.attn_pe == "relative":
            bucket = self._relative_bucket(q_pos[:, None] - kv_pos[None, :])
            emb = F.embedding(bucket, self.relative_bias)  # (Q, K, H)
            return emb.permute(2, 0, 1)
        if self.attn_pe == "alibi":
            dist = (q_pos[:, None] - kv_pos[None, :]).abs()
            return -self.alibi_slopes[:, None, None] * dist[None, :, :]
        if self.attn_pe == "lff":
            d = (q_pos[:, None] - kv_pos[None, :]).float()  # (Q, K)
            freq = self.lff_base_frequency * torch.exp(self.lff_log_scale)
            angle = d.unsqueeze(-1) * freq.unsqueeze(0)  # (Q, K, F)
            feat = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # (Q,K,2F)
            return torch.einsum("qkc,hc->hqk", feat, self.lff_proj)
        return None

    def forward(self, x, kv, q_pos, kv_pos):
        N, Q, D = x.shape
        K = kv.shape[1]
        H, dh = self.num_heads, self.head_dim
        q = self.q_proj(x).view(N, Q, H, dh).transpose(1, 2)
        k = self.k_proj(kv).view(N, K, H, dh).transpose(1, 2)
        v = self.v_proj(kv).view(N, K, H, dh).transpose(1, 2)
        if self.attn_pe == "rope":
            q = self._rope(q, q_pos)
            k = self._rope(k, kv_pos)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
        bias = self._bias(q_pos, kv_pos)
        if bias is not None:
            scores = scores + bias
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (N, H, Q, dh)
        out = out.transpose(1, 2).contiguous().view(N, Q, D)
        return self.o_proj(out), attn.detach()


class _ICPTEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, attn_pe, relative_buckets,
                 lff_frequencies, max_position, period_len, dropout):
        super().__init__()
        self.self_attn = _ICPTAttention(
            d_model, num_heads, attn_pe, relative_buckets, lff_frequencies,
            max_position, period_len, dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )

    def forward(self, x, positions):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), positions, positions)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class _ICPTDecoderBlock(nn.Module):
    """Decoder block: future-query self-attention then cross-attention to history."""

    def __init__(self, d_model, num_heads, ffn_dim, attn_pe, relative_buckets,
                 lff_frequencies, max_position, period_len, dropout):
        super().__init__()
        self.self_attn = _ICPTAttention(
            d_model, num_heads, attn_pe, relative_buckets, lff_frequencies,
            max_position, period_len, dropout,
        )
        self.cross_attn = _ICPTAttention(
            d_model, num_heads, attn_pe, relative_buckets, lff_frequencies,
            max_position, period_len, dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )
        self.last_cross_attn = None

    def forward(self, q, mem, q_pos, mem_pos):
        q = q + self.self_attn(self.norm1(q), self.norm1(q), q_pos, q_pos)[0]
        q, cross = self.cross_attn(self.norm2(q), self.norm2(mem), q_pos, mem_pos)
        self.last_cross_attn = cross
        q = q + self.ffn(self.norm3(q))
        return q


def _paired_sincos(angle):
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).flatten(-2)


class InterCyclePatchResidualHead(nn.Module):
    """Inter-cycle patch transformer residual head with pluggable position encoding.

    Input ``x`` is the RevIN-normalized ``(B, seq_len, C)`` history.  It is
    reshaped into ``(B, C, K_in, period_len)`` complete cycles, each cycle is
    tokenized by a linear patch projection, and an encoder/decoder transformer
    maps history cycle tokens to ``K_out`` future-cycle deltas relative to the
    most recent complete cycle (unless ``use_last_cycle_anchor=False``).  The
    output projection is zero-initialized so the initial output is exactly
    ``RepeatLastCycle``.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        d_model: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        pe_type: str = "none",
        relative_buckets: int = 16,
        lff_frequencies: int = 16,
        use_last_cycle_anchor: bool = True,
        use_attention: bool = True,
        label_len: int = 0,
        dropout: float = 0.0,
        prediction_head: str = "decoder",
        anchor_mode: str | None = None,
    ):
        super().__init__()
        if pe_type not in ICPTPE_TYPES:
            raise ValueError(f"Unsupported ICPT position encoding: {pe_type}")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if seq_len <= 0 or pred_len <= 0 or period_len <= 1:
            raise ValueError("seq_len/pred_len must be positive and period_len > 1")
        if prediction_head not in ("decoder", "flatten"):
            raise ValueError(f"Unsupported ICPT prediction head: {prediction_head}")
        if anchor_mode is None:
            anchor_mode = "last_cycle" if use_last_cycle_anchor else "none"
        if anchor_mode not in ("last_cycle", "last_value", "none"):
            raise ValueError(f"Unsupported ICPT anchor mode: {anchor_mode}")

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.d_model = int(d_model)
        self.pe_type = pe_type
        self.use_last_cycle_anchor = bool(use_last_cycle_anchor)
        self.use_attention = bool(use_attention)
        self.label_len = int(label_len)
        self.prediction_head = prediction_head
        self.anchor_mode = anchor_mode

        self.num_periods_input = (seq_len + period_len - 1) // period_len
        self.num_periods_output = (pred_len + period_len - 1) // period_len
        self.pad_input = self.num_periods_input * period_len - seq_len
        self.trim_output = self.num_periods_output * period_len - pred_len
        self.max_position = self.num_periods_input + (
            self.num_periods_output if prediction_head == "decoder" else 0
        )

        self.token_pe = _TOKEN_PE[pe_type]
        self.attn_pe = _ATTN_PE[pe_type]

        # Cycle token encoding: Linear(P -> d_model) + LayerNorm.
        self.patch_proj = nn.Linear(period_len, d_model)
        self.patch_norm = nn.LayerNorm(d_model)

        # Token-level position encoding parameters.
        if self.token_pe == "learned_abs":
            self.learned_abs = nn.Parameter(
                torch.zeros(self.max_position, d_model)
            )
            nn.init.trunc_normal_(self.learned_abs, std=0.02)
        elif self.token_pe == "time2vec":
            num_periodic = max(1, d_model - 1)
            harmonic = torch.arange(1, num_periodic + 1, dtype=torch.float32)
            harmonic = (harmonic - 1).remainder(8) + 1
            init_frequency = 2.0 * math.pi * harmonic / period_len
            self.time2vec_frequency = nn.Parameter(init_frequency)
            self.time2vec_phase = nn.Parameter(torch.zeros(num_periodic))
            self.time2vec_linear_weight = nn.Parameter(torch.ones(1))
            self.time2vec_linear_bias = nn.Parameter(torch.zeros(1))
        elif self.token_pe == "calendar":
            self.calendar_proj = nn.Linear(10, d_model)

        if prediction_head == "decoder":
            # Decoder query tokens (one per future cycle).
            self.learned_query = nn.Parameter(
                torch.randn(self.num_periods_output, d_model) * 0.02
            )

        if use_attention:
            block_kwargs = dict(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                attn_pe=self.attn_pe,
                relative_buckets=relative_buckets,
                lff_frequencies=lff_frequencies,
                max_position=self.max_position,
                period_len=period_len,
                dropout=dropout,
            )
            self.encoder_blocks = nn.ModuleList(
                [_ICPTEncoderBlock(**block_kwargs) for _ in range(encoder_layers)]
            )
            self.decoder_blocks = nn.ModuleList(
                [_ICPTDecoderBlock(**block_kwargs) for _ in range(decoder_layers)]
                if prediction_head == "decoder" else []
            )
        else:
            # B5: cycle-token MLP, no attention.  Flatten encoded cycle tokens and
            # predict all future-cycle deltas with an MLP (last layer zero-init).
            if prediction_head == "decoder":
                self.mlp_decoder = nn.Sequential(
                    nn.Linear(self.num_periods_input * d_model, ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, self.num_periods_output * period_len),
                )
                nn.init.zeros_(self.mlp_decoder[-1].weight)
                nn.init.zeros_(self.mlp_decoder[-1].bias)

        if prediction_head == "decoder":
            # Output projection: d_model -> period_len.  Zero init makes the
            # legacy decoder output exactly RepeatLastCycle.
            self.out_proj = nn.Linear(d_model, period_len)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            # PatchTST-style ordered full-horizon head. With d_model=P, its
            # main matrix has the same shape as NLinear's Linear(L, H).
            self.horizon_head = nn.Linear(
                self.num_periods_input * d_model, pred_len
            )
            nn.init.zeros_(self.horizon_head.weight)
            nn.init.zeros_(self.horizon_head.bias)

        # Diagnostics (no-grad captured, mirror the RCRF/PhaseVelocity hooks).
        self.last_attention = None
        self.last_attention_entropy = None
        self.last_top_lags = None
        self.last_delta_norm = None
        self.last_anchor_norm = None

    # ---- position encodings -------------------------------------------------
    def _sincos_pe(self, positions):
        d = self.d_model
        freqs = torch.pow(
            10000.0,
            -torch.arange(0, d, 2, device=positions.device, dtype=positions.dtype) / d,
        )
        angles = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (S, d/2)
        pe = torch.zeros(positions.size(0), d, device=positions.device, dtype=positions.dtype)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def _time2vec_pe(self, positions):
        scale = max(float(self.max_position - 1), 1.0)
        normalized = 2.0 * positions.float() / scale - 1.0
        linear = (
            normalized.unsqueeze(-1) * self.time2vec_linear_weight
            + self.time2vec_linear_bias
        )
        periodic = torch.sin(
            positions.float().unsqueeze(-1) * self.time2vec_frequency
            + self.time2vec_phase
        )
        return torch.cat((linear, periodic), dim=-1)

    def _token_pe(self, positions):
        # positions: (S,) absolute cycle indices; returns (S, d_model).
        if self.token_pe == "sincos":
            return self._sincos_pe(positions)
        if self.token_pe == "learned_abs":
            return self.learned_abs[positions]
        if self.token_pe == "time2vec":
            return self._time2vec_pe(positions)
        raise RuntimeError(f"{self.token_pe} is not a token-level PE")

    @staticmethod
    def _calendar_encode(marks, proj):
        # marks: (B, S, md); returns (B, S, d_model).
        if marks is None or marks.size(-1) < 4:
            raise ValueError("calendar PE requires raw timestamp marks with >=4 fields")
        month = marks[..., 0]
        day = marks[..., 1]
        weekday = marks[..., 2]
        hour = marks[..., 3]
        minute_slot = marks[..., 4] if marks.size(-1) >= 5 else torch.zeros_like(hour)
        time_of_day = (hour + minute_slot / 4.0) / 24.0
        cycles = torch.stack(
            (
                time_of_day,
                2.0 * time_of_day,
                weekday / 7.0,
                (day - 1.0) / 31.0,
                (month - 1.0) / 12.0,
            ),
            dim=-1,
        )
        feat = _paired_sincos(2.0 * math.pi * cycles)
        return proj(feat.float())

    def _calendar_pe(self, x_mark_enc, x_mark_dec):
        if x_mark_enc is None or x_mark_dec is None:
            raise ValueError(
                "calendar PE requires timestamp marks (x_mark_enc/x_mark_dec)"
            )
        # Cycle-start marks: history cycle k starts at encoder index k*P; future
        # cycle j starts at decoder index label_len + j*P.
        hist_idx = torch.arange(
            self.num_periods_input, device=x_mark_enc.device, dtype=torch.long
        ) * self.period_len
        fut_idx = torch.arange(
            self.num_periods_output, device=x_mark_dec.device, dtype=torch.long
        ) * self.period_len + self.label_len
        hist_marks = x_mark_enc[:, hist_idx, :]
        fut_marks = x_mark_dec[:, fut_idx, :]
        hist_pe = self._calendar_encode(hist_marks, self.calendar_proj)
        fut_pe = self._calendar_encode(fut_marks, self.calendar_proj)
        return hist_pe, fut_pe

    def _calendar_history_pe(self, x_mark_enc):
        if x_mark_enc is None:
            raise ValueError("calendar PE requires encoder timestamp marks")
        hist_idx = torch.arange(
            self.num_periods_input, device=x_mark_enc.device, dtype=torch.long
        ) * self.period_len
        return self._calendar_encode(x_mark_enc[:, hist_idx, :], self.calendar_proj)

    # ---- forward ------------------------------------------------------------
    def forward(self, x, x_mark_enc=None, x_mark_dec=None):
        # x: (B, seq_len, C) normalized scale.
        B, L, C = x.shape
        if self.pad_input > 0:
            x = F.pad(x, (0, 0, self.pad_input, 0), mode="replicate")
        xt = x.permute(0, 2, 1).contiguous()  # (B, C, K_in*P)
        X = xt.view(B, C, self.num_periods_input, self.period_len)
        cycle_anchor = X[:, :, -1, :]  # (B, C, P)
        value_anchor = xt[:, :, -1:]  # (B, C, 1)
        if self.anchor_mode == "last_cycle":
            M = X - cycle_anchor.unsqueeze(2)
        elif self.anchor_mode == "last_value":
            M = X - value_anchor.unsqueeze(2)
        else:
            M = X
        N = B * C
        z = self.patch_norm(self.patch_proj(M.reshape(N, self.num_periods_input, self.period_len)))

        hist_positions = torch.arange(self.num_periods_input, device=x.device, dtype=torch.long)
        fut_positions = self.num_periods_input + torch.arange(
            self.num_periods_output, device=x.device, dtype=torch.long
        )

        # Token-level PE.
        if self.token_pe == "calendar":
            if self.prediction_head == "decoder":
                hist_pe, fut_pe = self._calendar_pe(x_mark_enc, x_mark_dec)
            else:
                hist_pe = self._calendar_history_pe(x_mark_enc)
            # Calendar PE is per-sample; broadcast to every channel, then merge.
            hist_pe = hist_pe[:, None, :, :].expand(
                B, C, self.num_periods_input, self.d_model
            )
            z = z + hist_pe.reshape(N, self.num_periods_input, self.d_model)
            if self.prediction_head == "decoder":
                fut_pe = fut_pe[:, None, :, :].expand(
                    B, C, self.num_periods_output, self.d_model
                )
                q = self.learned_query.unsqueeze(0).expand(N, -1, -1) + fut_pe.reshape(
                    N, self.num_periods_output, self.d_model
                )
        elif self.token_pe != "none":
            z = z + self._token_pe(hist_positions)
            if self.prediction_head == "decoder":
                q = self.learned_query.unsqueeze(0).expand(N, -1, -1) + self._token_pe(
                    fut_positions
                )
        elif self.prediction_head == "decoder":
            q = self.learned_query.unsqueeze(0).expand(N, -1, -1)

        if self.use_attention:
            for block in self.encoder_blocks:
                z = block(z, hist_positions)
        if self.prediction_head == "flatten":
            delta = self.horizon_head(z.reshape(N, -1)).reshape(B, C, self.pred_len)
            self.last_attention = None
        elif self.use_attention:
            for block in self.decoder_blocks:
                q = block(q, z, fut_positions, hist_positions)
            delta = self.out_proj(q)  # (N, K_out, P)
            self._capture_attention(z, q, fut_positions, hist_positions)
        else:
            flat = z.reshape(N, self.num_periods_input * self.d_model)
            delta = self.mlp_decoder(flat).view(
                N, self.num_periods_output, self.period_len
            )
            self.last_attention = None

        if self.prediction_head == "flatten":
            if self.anchor_mode == "last_cycle":
                repeats = (self.pred_len + self.period_len - 1) // self.period_len
                base = cycle_anchor.repeat(1, 1, repeats)[:, :, : self.pred_len]
            elif self.anchor_mode == "last_value":
                base = value_anchor.expand(-1, -1, self.pred_len)
            else:
                base = torch.zeros_like(delta)
            y = (base + delta).permute(0, 2, 1)
        else:
            delta = delta.reshape(B, C, self.num_periods_output, self.period_len)
            if self.anchor_mode == "last_cycle":
                Y = cycle_anchor.unsqueeze(2) + delta
            elif self.anchor_mode == "last_value":
                Y = value_anchor.unsqueeze(2) + delta
            else:
                Y = delta
            y = Y.reshape(B, C, self.num_periods_output * self.period_len)
            if self.trim_output > 0:
                y = y[:, :, : self.pred_len]
            y = y.permute(0, 2, 1)  # (B, pred_len, C)

        with torch.no_grad():
            self.last_delta_norm = float(delta.detach().norm())
            anchor = cycle_anchor if self.anchor_mode == "last_cycle" else value_anchor
            self.last_anchor_norm = float(anchor.detach().norm())
        return y

    def _capture_attention(self, z, q, fut_positions, hist_positions):
        if not self.use_attention or not self.decoder_blocks:
            return
        attn = self.decoder_blocks[-1].last_cross_attn  # (N, H, K_out, K_in)
        if attn is None:
            return
        self.last_attention = attn
        mean_attn = attn.mean(dim=0).mean(dim=0)  # (K_out, K_in) averaged over N and heads
        self.last_attention_entropy = float(
            -(attn * attn.clamp_min(1e-12).log()).sum(dim=-1).mean().detach()
        )
        top_key = mean_attn.argmax(dim=-1)
        self.last_top_lags = (fut_positions - hist_positions[top_key]).detach()
