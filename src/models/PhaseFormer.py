import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.layers.SelfAttention_Family import AttentionLayer, FullAttention
from src.models.pl_bases.default_module import DefaultPLModule


from src.models.phase_adapters import (
    RevIN,
    WeakPeriodResidualHead,
    ChannelWiseWeakPeriodResidualHead,
    LowPassWeakPeriodResidualHead,
    AdaptiveWeakPeriodGate,
    TimeMarkAdjustmentHead,
    PhaseLocalTrendHead,
    PhaseUncertaintyShrinkage,
    PhasePeriodLevelCalibration,
    PhaseSparseEventCalibration,
    PhaseNoiseHighFreqDamping,
    PeriodPositionEncodedResidualHead,
    ReliabilityCoupledResidualFusion,
)
from src.models.periodic_residual_experts import (
    AdaptiveMultiPeriodResidualHead,
    DualReliabilityPeriodicFusion,
    PhaseErrorPeriodicMemoryHead,
)
from src.models.intercycle_patch import (
    CycleNetStyleResidualHead,
    InterCyclePatchResidualHead,
    RepeatLastCycleResidualHead,
)
from src.models.hierarchical_trend_cycle import (
    HierarchicalTrendCycleResidualHead,
)
from src.models.phase_cycle_trajectory import (
    PhaseCycleTrajectoryResidualHead,
)
from src.models.phase_cycle_fusion import PhaseCycleFusionComposer
from src.models.anchored_phase_cycle_fusion import (
    AnchoredPhaseCycleFusionComposer,
)
from src.models.triaxis_fusion import (
    RollingTriAxisHistoryRouter,
    SafeRegretTriAxisRouter,
    TriAxisHistoryRouter,
)
from src.models.phase_align import PhaseAlignment
from src.models.phase_warp import PhaseWarping
from src.models.phase_amp_calib import PhaseAmpCalibration
from src.models.phase_rape import ReliabilityGate
from src.models.phase_correction import PhaseCorrection
from src.models.phase_geometry import CircularPhaseEmbedding
from src.models.phase_rotation import PhaseRotation
from src.models.harmonic_modulation import HarmonicModulation
from src.models.phase_velocity import PhaseVelocity
from src.models.adaptive_residual_gate import AdaptiveResidualGate
from src.models.multiscale_phase import MultiScalePhase
from src.models.phase_deformation import PhaseDeformation
from src.models.phase_graph import PhaseGraph
from src.models.phase_decoder import TrajectoryDecoder
from src.models.residual_topology import (
    AdditiveOutputResidualHead,
    LatentResidualPath,
    PhaseSlotResidualHead,
)

class CrossPhaseRoutingLayer(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_routers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_relpos: bool = True,
        period_len: int = 24,
        window_size: Optional[int] = None,
        attention_dim: Optional[int] = None,
        use_pos_embed: bool = False,
        pos_dropout: float = 0.0,
        use_circular_pos: bool = False,
        use_circular_attn_bias: bool = False,
        circular_attn_bias_scale: float = 1.0,
    ):
        super().__init__()
        # The attention_dim parameter is kept for interface compatibility; it does not
        # change the projection dimensions in this implementation.
        self.attention_dim = attention_dim or latent_dim
        assert (
            self.attention_dim % num_heads == 0
        ), "attention_dim must be divisible by num_heads"

        self.latent_dim = latent_dim
        self.num_routers = num_routers
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_circular_pos = use_circular_pos
        self.use_circular_attn_bias = use_circular_attn_bias
        self.circular_attn_bias_scale = circular_attn_bias_scale
        self.period_len = period_len

        # Learnable routers shared across batch and channels
        self.router = nn.Parameter(torch.randn(num_routers, latent_dim))
        nn.init.trunc_normal_(self.router, std=0.02)

        # Optional phase positional embeddings (length equals period_len).
        # The learnable pos_embedding is always created when use_pos_embed is on
        # (also when circular geometry is enabled) so that toggling the circular
        # flag does not shift the RNG draws consumed by this layer; the circular
        # buffer below replaces it in forward() when use_circular_pos is set.
        if self.use_pos_embed:
            self.pos_embedding = nn.Parameter(torch.zeros(period_len, latent_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.pos_dropout = nn.Dropout(pos_dropout)
        if self.use_circular_pos:
            # Non-persistent buffer: no parameters, no RNG draws.
            self.circular_embedding = CircularPhaseEmbedding(period_len, latent_dim)

        # Two-stage attention: routers aggregate then distribute
        self.router_sender = AttentionLayer(
            FullAttention(
                False, factor=5, attention_dropout=dropout, output_attention=False
            ),
            latent_dim,
            num_heads,
        )
        self.router_receiver = AttentionLayer(
            FullAttention(
                False, factor=5, attention_dropout=dropout, output_attention=False
            ),
            latent_dim,
            num_heads,
        )

        # Post-attention residual + LayerNorm + MLP
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, Z):  # Z: (B, C, L, D)
        B, C, L, D = Z.shape
        x = Z.view(B * C, L, D)

        # Optional positional embedding. When circular geometry is enabled the
        # learnable embedding is replaced by the fixed circular (Fourier) phase
        # embedding; the learnable parameters are still created at construction
        # (see __init__) purely to keep flag-off initialization identical.
        if self.use_pos_embed:
            pe_source = (
                self.circular_embedding.embedding
                if self.use_circular_pos
                else self.pos_embedding
            )
            if L == self.period_len:
                pe = pe_source.unsqueeze(0).expand(B * C, -1, -1)
            elif L < self.period_len:
                pe = pe_source[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            else:
                repeat_factor = (L + self.period_len - 1) // self.period_len
                expanded_pe = pe_source.repeat(repeat_factor, 1)
                pe = expanded_pe[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            x = x + pe
            x = self.pos_dropout(x)

        # Circular attention bias (next-stage plan stage 2): anchors each router
        # at a canonical phase position r * P / R on the phase circle and
        # penalizes attention to phase slots far from it, so the interaction
        # layer itself (not just the position embedding) becomes cycle-aware.
        # The bias is deterministic from (R, P, L): no parameters, no RNG draws.
        if self.use_circular_attn_bias:
            r_pos = torch.arange(
                self.num_routers, device=Z.device, dtype=torch.float32
            ) * (self.period_len / self.num_routers)
            slot_pos = torch.arange(L, device=Z.device, dtype=torch.float32)
            raw = (r_pos.unsqueeze(1) - slot_pos.unsqueeze(0)).abs()  # (R, L)
            dist = torch.minimum(raw, self.period_len - raw)  # circular distance
            bias = self.circular_attn_bias_scale * (dist / (self.period_len / 2))
            bias_sender = bias.unsqueeze(0).unsqueeze(0)  # (1, 1, R, L)
            bias_receiver = bias.t().unsqueeze(0).unsqueeze(0)  # (1, 1, L, R)
        else:
            bias_sender = None
            bias_receiver = None

        # Stage 1: routers aggregate token information
        batch_router = self.router.unsqueeze(0).expand(B * C, -1, -1)  # (BC, R, D)
        router_buffer, _ = self.router_sender(
            batch_router, x, x, attn_mask=None, bias=bias_sender
        )

        # Stage 2: routers distribute information back to tokens
        router_receive, _ = self.router_receiver(
            x, router_buffer, router_buffer, attn_mask=None, bias=bias_receiver
        )

        # Residual + LayerNorm
        out = x + self.dropout_layer(router_receive)
        out = self.norm1(out)

        # MLP block + Residual + LayerNorm
        mlp_out = self.mlp(out)
        out = out + self.dropout_layer(mlp_out)
        out = self.norm2(out)

        # Restore shape back to (B, C, L, D)
        out = out.view(B, C, L, D)
        return out



class PhaseEmbedding(nn.Module):
    """Projects phase-series tokens (P_in) into the latent space (D) with optional MLP.

    This layer applies a linear projection (or small MLP) across the phase dimension
    and then normalizes with LayerNorm to stabilize training.
    """
    def __init__(
        self,
        p_in: int,
        latent_dim: int,
        hidden: int = 32,
        use_mlp: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_mlp = use_mlp
        self.norm = nn.LayerNorm(latent_dim)
        if use_mlp:
            self.projection = nn.Sequential(
                nn.Linear(p_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, latent_dim),
            )
        else:
            self.projection = nn.Linear(p_in, latent_dim)

    def forward(self, phase_series):  # (B, C, L, P_in)
        return self.norm(self.projection(phase_series))




class PhasePredictor(nn.Module):
    """Maps latent features to the desired number of output phase steps (P_out).

    By default this is a single linear layer with optional dropout. An optional
    small MLP can be enabled via configuration, but the default matches the
    original implementation (use_mlp=False).
    """
    
    def __init__(self, p_out: int, latent_dim: int, hidden: int, use_mlp: bool = False, dropout: float = 0.0):
        super().__init__()
        self.p_out = p_out
        self.use_mlp = use_mlp
        
        if use_mlp:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden, p_out)
            )
        else:
            self.decoder = nn.Linear(latent_dim, p_out)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self, z):
        """
        Input: z (B, C, L, latent_dim)
        Output: (B, C, L, p_out)
        """
        if self.use_mlp:
            return self.decoder(z)  # (B, C, L, p_out)
        else:
            z = self.dropout(z)
            return self.decoder(z)  # (B, C, L, p_out)


class CrossPhaseRoutingUnit(nn.Module):
    """Routing unit with optional linear in/out projections around cross-phase routing.

    Composition to preserve the original information flow:
    - First unit: apply_in_proj=False, apply_out_proj=True (produce P_in for next layer)
    - Middle units: apply_in_proj=True, apply_out_proj=True
    - Last unit: apply_in_proj=True, apply_out_proj=False (final P_out via top-level predictor)
    """

    def __init__(
        self,
        *,
        apply_in_proj: bool,
        apply_out_proj: bool,
        num_periods_input: int,
        latent_dim: int,
        phase_encoder_hidden: int,
        predictor_hidden: int,
        phase_attn_heads: int,
        phase_attn_dropout: float,
        phase_attn_use_relpos: bool,
        period_len: int,
        phase_attn_window=None,
        phase_attention_dim=None,
        phase_num_routers: int = 8,
        phase_use_pos_embed: bool = False,
        phase_pos_dropout: float = 0.0,
        phase_use_circular_pos: bool = False,
        phase_use_circular_attn_bias: bool = False,
        phase_circular_attn_bias_scale: float = 1.0,
        phase_encoder_use_mlp: bool = False,
        phase_encoder_dropout: float = 0.0,
        predictor_use_mlp: bool = False,
        predictor_dropout: float = 0.0,
    ):
        super().__init__()
        self.apply_in_proj = apply_in_proj
        self.apply_out_proj = apply_out_proj

        if self.apply_in_proj:
            # Linear in-projection from P_in to latent_dim; include LayerNorm to match PhaseEmbedding(use_mlp=False)
            self.in_proj = nn.Sequential(
                nn.Linear(num_periods_input, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.in_proj = None

        self.interact = CrossPhaseRoutingLayer(
            latent_dim=latent_dim,
            num_routers=phase_num_routers,
            num_heads=phase_attn_heads,
            dropout=phase_attn_dropout,
            use_relpos=phase_attn_use_relpos,
            period_len=period_len,
            window_size=phase_attn_window,
            attention_dim=phase_attention_dim,
            use_pos_embed=phase_use_pos_embed,
            pos_dropout=phase_pos_dropout,
            use_circular_pos=phase_use_circular_pos,
            use_circular_attn_bias=phase_use_circular_attn_bias,
            circular_attn_bias_scale=phase_circular_attn_bias_scale,
        )

        if self.apply_out_proj:
            # Linear out-projection back to P_in for chaining to the next layer
            self.out_proj = nn.Linear(latent_dim, num_periods_input)
        else:
            self.out_proj = None

    def forward(self, phase_series, z_prev=None):
        # Inputs:
        #   phase_series: (B, C, L, P_in)
        #   z_prev: (B, C, L, D) or None (must be provided if apply_in_proj is False)
        if self.apply_in_proj:
            z_curr = self.in_proj(phase_series)
            if z_prev is not None:
                z = z_prev + z_curr
            else:
                z = z_curr
        else:
            assert z_prev is not None, "z_prev must be provided when apply_in_proj is False"
            z = z_prev

        z = self.interact(z)

        if self.out_proj is not None:
            y_phase_steps = self.out_proj(z)  # (B, C, L, P_in)
        else:
            y_phase_steps = None

        return z, y_phase_steps


class PhaseFormerBlock(nn.Module):
    """Legacy block kept for reference and minimal disruption of imports.

    It represents a single layer of the original design: Encoder -> Interaction -> Decoder.
    The new implementation uses top-level embedding and predictor with routing units
    in between. This class is unused in the new data path but preserved to avoid
    breaking external references.
    """

    def __init__(
        self,
        num_periods_input: int,
        num_periods_output: int,
        latent_dim: int,
        phase_encoder_hidden: int,
        predictor_hidden: int,
        phase_attn_heads: int,
        phase_attn_dropout: float,
        phase_attn_use_relpos: bool,
        period_len: int,
        phase_attn_window=None,
        phase_attention_dim=None,
        phase_num_routers: int = 8,
        phase_use_pos_embed: bool = False,
        phase_pos_dropout: float = 0.0,
        phase_encoder_use_mlp: bool = False,
        phase_encoder_dropout: float = 0.0,
        predictor_use_mlp: bool = False,
        predictor_dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = PhaseEmbedding(
            p_in=num_periods_input,
            latent_dim=latent_dim,
            hidden=phase_encoder_hidden,
            use_mlp=phase_encoder_use_mlp,
            dropout=phase_encoder_dropout,
        )

        self.interact = CrossPhaseRoutingLayer(
            latent_dim=latent_dim,
            num_routers=phase_num_routers,
            num_heads=phase_attn_heads,
            dropout=phase_attn_dropout,
            use_relpos=phase_attn_use_relpos,
            period_len=period_len,
            window_size=phase_attn_window,
            attention_dim=phase_attention_dim,
            use_pos_embed=phase_use_pos_embed,
            pos_dropout=phase_pos_dropout,
        )

        self.decoder = PhasePredictor(
            p_out=num_periods_output,
            latent_dim=latent_dim,
            hidden=predictor_hidden,
            use_mlp=predictor_use_mlp,
            dropout=predictor_dropout,
        )

    def forward(self, phase_series, z_prev=None):
        # phase_series: (B, C, L, P_in)
        # z_prev: (B, C, L, D) or None
        z_curr = self.encoder(phase_series)  # (B, C, L, D)
        if z_prev is not None:
            # residual aggregation across layers
            z = z_prev + z_curr
        else:
            z = z_curr

        z = self.interact(z)
        y_phase_steps = self.decoder(z)  # (B, C, L, P_out)
        return z, y_phase_steps


class PhaseFormer(DefaultPLModule):
    """
    PhaseFormer: phase-based modeling without cross-channel fusion.

    Pipeline:
    1) RevIN over time per variable
    2) Embedding -> [CrossPhaseRouting] x N -> Predictor produce future phase steps
    3) Reassemble to forecasting sequence and de-normalize
    """

    def __init__(self, configs):
        super().__init__(configs)

        # basic configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        # phase dimensions
        self.latent_dim = getattr(configs, "latent_dim", 8)
        self.phase_encoder_hidden = getattr(configs, "phase_encoder_hidden", 32)
        self.predictor_hidden = getattr(configs, "predictor_hidden", 64)

        # phase attention parameters
        self.phase_attn_heads = getattr(configs, "phase_attn_heads", 4)
        self.phase_attn_dropout = getattr(configs, "phase_attn_dropout", 0.0)
        self.phase_attn_use_relpos = getattr(configs, "phase_attn_use_relpos", True)
        self.phase_attn_window = getattr(configs, "phase_attn_window", None)
        self.phase_attention_dim = getattr(configs, "phase_attention_dim", None)
        self.phase_num_routers = getattr(configs, "phase_num_routers", 8)
        self.phase_use_pos_embed = getattr(configs, "phase_use_pos_embed", False)
        self.phase_pos_dropout = getattr(configs, "phase_pos_dropout", 0.0)
        self.phase_use_circular_pos = getattr(configs, "phase_use_circular_pos", False)
        self.phase_use_circular_attn_bias = getattr(
            configs, "phase_use_circular_attn_bias", False
        )
        self.phase_circular_attn_bias_scale = getattr(
            configs, "phase_circular_attn_bias_scale", 1.0
        )

        # period calculations
        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        # RevIN normalization
        self.use_revin = getattr(configs, "use_revin", True)
        self.revin_affine = getattr(configs, "revin_affine", False)
        self.revin_eps = getattr(configs, "revin_eps", 1e-5)
        if self.use_revin:
            self.revin = RevIN(num_features=self.enc_in, eps=self.revin_eps, affine=self.revin_affine)

        # Residual-branch master switch (experiment plan stage 1): when disabled,
        # the WeakPeriodResidualHead and PhaseLocalTrendHead are both turned off,
        # i.e. the model predicts purely from the phase path with no residual branch.
        self.use_residual_head = getattr(configs, "use_residual_head", True)
        if not self.use_residual_head:
            self.use_weak_period_residual = False

        self.use_weak_period_residual = getattr(configs, "use_weak_period_residual", False)
        if not self.use_residual_head:
            self.use_weak_period_residual = False
        self.use_phase_cycle_fusion = getattr(
            configs, "use_phase_cycle_fusion", False
        )
        self.use_anchored_phase_cycle_fusion = getattr(
            configs, "use_anchored_phase_cycle_fusion", False
        )
        if not self.use_residual_head:
            self.use_phase_cycle_fusion = False
            self.use_anchored_phase_cycle_fusion = False
        self.use_triaxis_fusion = getattr(configs, "use_triaxis_fusion", False)
        self.use_safe_triaxis = getattr(configs, "use_safe_triaxis", False)
        if not self.use_residual_head:
            self.use_triaxis_fusion = False
            self.use_safe_triaxis = False
        if self.use_triaxis_fusion and self.use_weak_period_residual:
            raise ValueError(
                "TriAxis owns its atomic trajectory/cycle experts and cannot be "
                "combined with the legacy weak residual branch"
            )
        if self.use_triaxis_fusion and self.use_safe_triaxis:
            raise ValueError("legacy TriAxis and Safe-Regret TriAxis are exclusive")
        if self.use_phase_cycle_fusion and (
            self.use_weak_period_residual
            or self.use_triaxis_fusion
            or self.use_safe_triaxis
            or self.use_anchored_phase_cycle_fusion
        ):
            raise ValueError(
                "phase-cycle fusion owns its three branches and is exclusive "
                "with weak residual and TriAxis paths"
            )
        if self.use_anchored_phase_cycle_fusion and (
            self.use_triaxis_fusion or self.use_safe_triaxis
        ):
            raise ValueError(
                "A2-anchored phase-cycle fusion is exclusive with TriAxis paths"
            )
        self.use_adaptive_weak_period_gate = getattr(
            configs, "use_adaptive_weak_period_gate", False
        )
        # RCRF (Reliability-Coupled Residual Fusion) replaces the fixed/adaptive
        # gate with a reliability-coupled convex gate. It is mutually exclusive
        # with the adaptive gates: when enabled, the adaptive gate modules are not
        # constructed and forward routes exclusively through RCRF.
        self.use_rcrf_fusion = getattr(configs, "use_rcrf_fusion", False)
        self.use_periodic_residual_pe = getattr(
            configs, "use_periodic_residual_pe", False
        )
        self.use_dual_reliability_fusion = getattr(
            configs, "use_dual_reliability_fusion", False
        )
        self.intercycle_head_requires_marks = False
        if self.use_phase_cycle_fusion:
            self.phase_cycle_fusion = PhaseCycleFusionComposer(
                self.seq_len,
                self.pred_len,
                self.period_len,
                strategy=getattr(
                    configs, "phase_cycle_fusion_strategy", "component_cycle"
                ),
                d_model=getattr(configs, "phase_cycle_fusion_d_model", 32),
                num_heads=getattr(configs, "phase_cycle_fusion_heads", 4),
                ffn_dim=getattr(configs, "phase_cycle_fusion_ffn_dim", 64),
                level_gate_init=getattr(
                    configs, "phase_cycle_fusion_level_gate_init", 0.10
                ),
                shape_gate_init=getattr(
                    configs, "phase_cycle_fusion_shape_gate_init", 0.10
                ),
                deformation_gate_init=getattr(
                    configs, "phase_cycle_fusion_deformation_gate_init", 0.05
                ),
                masked_origins=getattr(
                    configs, "phase_cycle_fusion_masked_origins", 2
                ),
                risk_scale=getattr(
                    configs, "phase_cycle_fusion_risk_scale", 1.0
                ),
                risk_std_weight=getattr(
                    configs, "phase_cycle_fusion_risk_std_weight", 0.5
                ),
                confidence_floor=getattr(
                    configs, "phase_cycle_fusion_confidence_floor", 0.05
                ),
                evidence_strength_init=getattr(
                    configs, "phase_cycle_fusion_evidence_strength_init", 1.0
                ),
                mlp_hidden=getattr(
                    configs, "phase_cycle_fusion_mlp_hidden", 16
                ),
                mlp_correction_max=getattr(
                    configs, "phase_cycle_fusion_mlp_correction_max", 2.0
                ),
                modulation_temperature=getattr(
                    configs, "phase_cycle_fusion_modulation_temperature", 0.10
                ),
                amplitude_min=getattr(
                    configs, "phase_cycle_fusion_amplitude_min", 0.5
                ),
                amplitude_max=getattr(
                    configs, "phase_cycle_fusion_amplitude_max", 2.0
                ),
            )
        # Ablation-2 mode: no A2 branch exists.  The composer corrects
        # PhaseFormer's own forecast in place, and the trajectory residual
        # path (weak period residual / RCRF / periodic-PE) must be absent.
        # Parsed before the guard because the guard branches on it.
        self.anchored_pctf_anchorless = bool(getattr(
            configs, "anchored_pctf_anchorless", False
        ))
        if self.use_anchored_phase_cycle_fusion:
            if self.anchored_pctf_anchorless:
                # Ablation-2 anchorless mode requires the complete absence of
                # the A2 trajectory path; the composer anchors on PhaseFormer's
                # own forecast instead.
                if (
                    self.use_weak_period_residual
                    or self.use_rcrf_fusion
                    or self.use_periodic_residual_pe
                ):
                    raise ValueError(
                        "anchored_pctf_anchorless requires the A2 trajectory "
                        "path (weak period residual / RCRF / periodic-PE) disabled"
                    )
            else:
                if not self.use_weak_period_residual or not self.use_rcrf_fusion:
                    raise ValueError(
                        "anchored phase-cycle fusion requires the complete RCRF "
                        "residual anchor"
                    )
                if not self.use_periodic_residual_pe:
                    raise ValueError(
                        "anchored phase-cycle fusion requires the A2 periodic-PE "
                        "trajectory branch"
                    )
                if getattr(configs, "periodic_residual_pe_type", None) != "lff":
                    raise ValueError(
                        "anchored phase-cycle fusion is defined relative to the "
                        "rcrf_pe_lff A2 anchor"
                    )
            self.anchored_phase_cycle_fusion = AnchoredPhaseCycleFusionComposer(
                self.seq_len,
                self.pred_len,
                cycle_period_len=getattr(
                    configs, "anchored_pctf_cycle_period_len", self.period_len
                ),
                strategy=getattr(
                    configs, "anchored_pctf_strategy", "component_cycle"
                ),
                d_model=getattr(configs, "anchored_pctf_d_model", 32),
                num_heads=getattr(configs, "anchored_pctf_heads", 4),
                ffn_dim=getattr(configs, "anchored_pctf_ffn_dim", 64),
                correction_max=getattr(
                    configs, "anchored_pctf_correction_max", 0.25
                ),
                deformation_max=getattr(
                    configs, "anchored_pctf_deformation_max", 0.10
                ),
                masked_origins=getattr(
                    configs, "anchored_pctf_masked_origins", 3
                ),
                risk_scale=getattr(configs, "anchored_pctf_risk_scale", 1.0),
                risk_std_weight=getattr(
                    configs, "anchored_pctf_risk_std_weight", 0.5
                ),
                confidence_floor=getattr(
                    configs, "anchored_pctf_confidence_floor", 0.05
                ),
                risk_clip=getattr(configs, "anchored_pctf_risk_clip", 6.0),
                mlp_hidden=getattr(configs, "anchored_pctf_mlp_hidden", 16),
                modulation_temperature=getattr(
                    configs, "anchored_pctf_modulation_temperature", 0.25
                ),
                amplitude_min=getattr(
                    configs, "anchored_pctf_amplitude_min", 0.5
                ),
                amplitude_max=getattr(
                    configs, "anchored_pctf_amplitude_max", 2.0
                ),
                detach_references=getattr(
                    configs, "anchored_pctf_detach_references", False
                ),
                level_mode=getattr(
                    configs, "anchored_pctf_level_mode", "horizon_centered"
                ),
                global_level_max=getattr(
                    configs, "anchored_pctf_global_level_max", 0.05
                ),
            )
            self.anchored_pctf_shape_aux_weight = float(getattr(
                configs, "anchored_pctf_shape_aux_weight", 0.05
            ))
            self.anchored_pctf_level_aux_weight = float(getattr(
                configs, "anchored_pctf_level_aux_weight", 0.05
            ))
            if (
                self.anchored_pctf_shape_aux_weight < 0
                or self.anchored_pctf_level_aux_weight < 0
            ):
                raise ValueError("anchored PCTF auxiliary weights must be non-negative")
            self.anchored_pctf_aux_target = str(getattr(
                configs, "anchored_pctf_aux_target", "absolute"
            ))
            if self.anchored_pctf_aux_target not in ("absolute", "residual"):
                raise ValueError("anchored PCTF auxiliary target is invalid")
            self.anchored_pctf_anchor_loss_weight = float(getattr(
                configs, "anchored_pctf_anchor_loss_weight", 0.0
            ))
            self.anchored_pctf_gate_aux_weight = float(getattr(
                configs, "anchored_pctf_gate_aux_weight", 0.0
            ))
            self.anchored_pctf_freeze_anchor = bool(getattr(
                configs, "anchored_pctf_freeze_anchor", False
            ))
            self.anchored_pctf_anchor_lr_scale = float(getattr(
                configs, "anchored_pctf_anchor_lr_scale", 1.0
            ))
            self.anchored_pctf_composer_lr_scale = float(getattr(
                configs, "anchored_pctf_composer_lr_scale", 1.0
            ))
            self.anchored_pctf_correction_warmup_epochs = int(getattr(
                configs, "anchored_pctf_correction_warmup_epochs", 0
            ))
            self.anchored_pctf_decouple_anchor_gradient = bool(getattr(
                configs, "anchored_pctf_decouple_anchor_gradient", False
            ))
            self.anchored_pctf_detach_composer_inputs = bool(getattr(
                configs, "anchored_pctf_detach_composer_inputs", False
            ))
            if (
                self.anchored_pctf_anchor_loss_weight < 0
                or self.anchored_pctf_gate_aux_weight < 0
                or not 0.0 <= self.anchored_pctf_anchor_lr_scale <= 1.0
                or self.anchored_pctf_composer_lr_scale <= 0.0
                or self.anchored_pctf_correction_warmup_epochs < 0
            ):
                raise ValueError("invalid anchored PCTF optimization settings")
            if self.anchored_pctf_correction_warmup_epochs:
                self.anchored_phase_cycle_fusion.set_correction_scale(0.0)
            self.anchored_pctf_anchor_output = None
            self.anchored_pctf_cycle_output = None
            self.anchored_pctf_level_correction_output = None
            self.anchored_pctf_shape_correction_output = None
            self.anchored_pctf_last_gate_aux_loss = None
        if self.use_weak_period_residual:
            residual_head_type = getattr(configs, "weak_period_residual_head_type", "shared")
            if self.use_dual_reliability_fusion and not self.use_rcrf_fusion:
                raise ValueError("dual reliability fusion requires use_rcrf_fusion=True")
            if (
                self.use_dual_reliability_fusion
                and residual_head_type != "periodic_pe"
            ):
                raise ValueError(
                    "dual reliability fusion requires the periodic_pe residual head"
                )
            if residual_head_type == "periodic_pe":
                if not self.use_periodic_residual_pe:
                    raise ValueError(
                        "weak_period_residual_head_type=periodic_pe requires "
                        "use_periodic_residual_pe=True"
                    )
                self.weak_period_residual = PeriodPositionEncodedResidualHead(
                    self.seq_len,
                    self.pred_len,
                    self.period_len,
                    encoding_type=getattr(
                        configs, "periodic_residual_pe_type", "harmonic"
                    ),
                    pe_dim=getattr(configs, "periodic_residual_pe_dim", 16),
                    temperature=getattr(
                        configs, "periodic_residual_pe_temperature", 0.1
                    ),
                    cycle_decay=getattr(
                        configs, "periodic_residual_pe_cycle_decay", 0.1
                    ),
                    blend_init=getattr(
                        configs, "periodic_residual_pe_blend_init", 0.1
                    ),
                    learn_blend=not self.use_dual_reliability_fusion,
                )
            elif residual_head_type == "intercycle":
                self.weak_period_residual = InterCyclePatchResidualHead(
                    seq_len=self.seq_len,
                    pred_len=self.pred_len,
                    period_len=getattr(configs, "intercycle_period_len", 24),
                    d_model=getattr(configs, "intercycle_d_model", 32),
                    num_heads=getattr(configs, "intercycle_heads", 4),
                    ffn_dim=getattr(configs, "intercycle_ffn_dim", 64),
                    encoder_layers=getattr(configs, "intercycle_encoder_layers", 1),
                    decoder_layers=getattr(configs, "intercycle_decoder_layers", 1),
                    pe_type=getattr(configs, "intercycle_pe_type", "none"),
                    relative_buckets=getattr(
                        configs, "intercycle_relative_buckets", 16
                    ),
                    lff_frequencies=getattr(configs, "intercycle_lff_frequencies", 16),
                    use_last_cycle_anchor=getattr(
                        configs, "intercycle_use_last_cycle_anchor", True
                    ),
                    use_attention=getattr(configs, "intercycle_use_attention", True),
                    label_len=getattr(configs, "label_len", 0),
                    dropout=getattr(configs, "intercycle_dropout", 0.0),
                    prediction_head=getattr(
                        configs, "intercycle_prediction_head", "decoder"
                    ),
                    anchor_mode=getattr(
                        configs, "intercycle_anchor_mode", None
                    ),
                )
                # Calendar PE reads timestamp marks already provided to the model.
                self.intercycle_head_requires_marks = self.weak_period_residual.pe_type == "calendar"
            elif residual_head_type == "repeat_last_cycle":
                self.weak_period_residual = RepeatLastCycleResidualHead(
                    self.seq_len, self.pred_len, self.period_len
                )
            elif residual_head_type == "cycle_net":
                self.weak_period_residual = CycleNetStyleResidualHead(
                    self.seq_len, self.pred_len, self.period_len
                )
            elif residual_head_type == "phase_error_memory":
                self.weak_period_residual = PhaseErrorPeriodicMemoryHead(
                    self.seq_len,
                    self.pred_len,
                    self.period_len,
                    memory_dim=getattr(configs, "phase_error_memory_dim", 16),
                    temperature=getattr(
                        configs, "phase_error_memory_temperature", 0.1
                    ),
                    recency_decay=getattr(
                        configs, "phase_error_memory_recency_decay", 0.1
                    ),
                    max_correction=getattr(
                        configs, "phase_error_memory_max_correction", 0.5
                    ),
                )
            elif residual_head_type == "adaptive_multiperiod":
                self.weak_period_residual = AdaptiveMultiPeriodResidualHead(
                    self.seq_len,
                    self.pred_len,
                    periods=getattr(
                        configs, "multiperiod_residual_periods", (12, 24, 48, 96)
                    ),
                    routing_temperature=getattr(
                        configs, "multiperiod_residual_temperature", 0.15
                    ),
                    recency_decay=getattr(
                        configs, "multiperiod_residual_recency_decay", 0.1
                    ),
                    max_correction=getattr(
                        configs, "multiperiod_residual_max_correction", 0.5
                    ),
                )
            elif residual_head_type == "hierarchical_trend_cycle":
                self.weak_period_residual = HierarchicalTrendCycleResidualHead(
                    self.seq_len,
                    self.pred_len,
                    self.period_len,
                    d_model=getattr(configs, "hptc_d_model", 32),
                    num_heads=getattr(configs, "hptc_heads", 4),
                    ffn_dim=getattr(configs, "hptc_ffn_dim", 64),
                    beta_init=getattr(configs, "hptc_beta_init", 0.25),
                    use_rolling_confidence=getattr(
                        configs, "hptc_use_rolling_confidence", True
                    ),
                    rolling_origins=getattr(configs, "hptc_rolling_origins", 4),
                    recency_decay=getattr(configs, "hptc_recency_decay", 0.5),
                    risk_scale=getattr(configs, "hptc_risk_scale", 1.0),
                    risk_std_weight=getattr(
                        configs, "hptc_risk_std_weight", 1.0
                    ),
                    confidence_floor=getattr(
                        configs, "hptc_confidence_floor", 0.05
                    ),
                )
            elif residual_head_type == "phase_cycle_trajectory":
                self.weak_period_residual = PhaseCycleTrajectoryResidualHead(
                    self.seq_len,
                    self.pred_len,
                    self.period_len,
                    d_model=getattr(configs, "pctf_d_model", 32),
                    num_heads=getattr(configs, "pctf_heads", 4),
                    ffn_dim=getattr(configs, "pctf_ffn_dim", 64),
                    shape_gate_init=getattr(
                        configs, "pctf_shape_gate_init", 0.10
                    ),
                    level_gate_init=getattr(
                        configs, "pctf_level_gate_init", 0.10
                    ),
                    use_shape_correction=getattr(
                        configs, "pctf_use_shape_correction", True
                    ),
                    use_level_correction=getattr(
                        configs, "pctf_use_level_correction", True
                    ),
                    confidence_mode=getattr(
                        configs, "pctf_confidence_mode", "fixed"
                    ),
                    masked_origins=getattr(
                        configs, "pctf_masked_origins", 2
                    ),
                    risk_scale=getattr(configs, "pctf_risk_scale", 1.0),
                    risk_std_weight=getattr(
                        configs, "pctf_risk_std_weight", 0.5
                    ),
                    confidence_floor=getattr(
                        configs, "pctf_confidence_floor", 0.05
                    ),
                    risk_clip=getattr(configs, "pctf_risk_clip", 10.0),
                )
            elif residual_head_type == "channel":
                self.weak_period_residual = ChannelWiseWeakPeriodResidualHead(
                    self.seq_len, self.pred_len, self.enc_in
                )
            elif residual_head_type == "lowpass":
                self.weak_period_residual = LowPassWeakPeriodResidualHead(
                    self.seq_len,
                    self.pred_len,
                    window=getattr(configs, "weak_period_residual_smooth_window", 25),
                )
            else:
                self.weak_period_residual = WeakPeriodResidualHead(
                    self.seq_len, self.pred_len
                )
            gate_init = float(getattr(configs, "weak_period_residual_gate_init", 0.2))
            if self.use_rcrf_fusion:
                if self.use_dual_reliability_fusion:
                    self.dual_reliability_fusion = DualReliabilityPeriodicFusion(
                        pred_len=self.pred_len,
                        alpha_init=getattr(configs, "rcrf_alpha_init", 0.5),
                        phase_sensitivity_init=getattr(
                            configs, "rcrf_sensitivity_init", 0.0
                        ),
                        phase_s_max=getattr(configs, "rcrf_s_max", 4.0),
                        periodic_init=getattr(
                            configs, "dual_reliability_periodic_init", 0.1
                        ),
                        periodic_sensitivity_init=getattr(
                            configs, "dual_reliability_sensitivity_init", 2.0
                        ),
                        periodic_s_max=getattr(
                            configs, "dual_reliability_s_max", 4.0
                        ),
                        eps=getattr(configs, "rcrf_eps", 1e-6),
                    )
                else:
                    self.rcrf_fusion = ReliabilityCoupledResidualFusion(
                        alpha_init=getattr(configs, "rcrf_alpha_init", 0.5),
                        sensitivity_init=getattr(configs, "rcrf_sensitivity_init", 0.0),
                        s_max=getattr(configs, "rcrf_s_max", 4.0),
                        eps=getattr(configs, "rcrf_eps", 1e-6),
                    )
            elif self.use_adaptive_weak_period_gate:
                self.adaptive_weak_period_gate = AdaptiveWeakPeriodGate(
                    enc_in=self.enc_in,
                    hidden=getattr(configs, "adaptive_weak_period_gate_hidden", 8),
                    gate_init=gate_init,
                )
            else:
                gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
                gate_logit = torch.logit(torch.tensor(gate_init))
                self.weak_period_residual_gate = nn.Parameter(
                    torch.full((1, 1, self.enc_in), float(gate_logit))
                )

        if self.use_triaxis_fusion:
            cycle_period = getattr(configs, "triaxis_cycle_period_len", 24)
            if cycle_period != self.period_len:
                raise ValueError(
                    "TriAxis phase and cycle experts must use the same period_len"
                )
            self.triaxis_trajectory_expert = WeakPeriodResidualHead(
                self.seq_len, self.pred_len
            )
            self.triaxis_cycle_expert = InterCyclePatchResidualHead(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                period_len=cycle_period,
                d_model=getattr(configs, "triaxis_cycle_d_model", 32),
                num_heads=getattr(configs, "triaxis_cycle_heads", 4),
                ffn_dim=getattr(configs, "triaxis_cycle_ffn_dim", 64),
                encoder_layers=getattr(
                    configs, "triaxis_cycle_encoder_layers", 1
                ),
                decoder_layers=getattr(
                    configs, "triaxis_cycle_decoder_layers", 1
                ),
                pe_type="none",
                relative_buckets=16,
                lff_frequencies=16,
                use_last_cycle_anchor=True,
                use_attention=True,
                label_len=getattr(configs, "label_len", 0),
                dropout=0.0,
                prediction_head="decoder",
                anchor_mode="last_cycle",
            )
            router_mode = getattr(
                configs, "triaxis_router_mode", "self_validating"
            )
            if getattr(configs, "triaxis_router_family", "single_cutoff") == "rolling":
                self.triaxis_router = RollingTriAxisHistoryRouter(
                    pred_len=self.pred_len,
                    period_len=self.period_len,
                    mode=router_mode,
                    hidden=getattr(configs, "triaxis_router_hidden", 16),
                    origins=getattr(configs, "triaxis_rolling_origins", 4),
                    trajectory_window_cycles=getattr(
                        configs, "triaxis_trajectory_window_cycles", 4
                    ),
                    recency_decay=getattr(
                        configs, "triaxis_rolling_recency_decay", 0.5
                    ),
                    risk_prior_strength=getattr(
                        configs, "triaxis_risk_prior_strength", 1.0
                    ),
                    correction_max=getattr(
                        configs, "triaxis_router_correction_max", 0.5
                    ),
                    temperature=getattr(
                        configs, "triaxis_router_temperature", 1.0
                    ),
                )
            else:
                self.triaxis_router = TriAxisHistoryRouter(
                    pred_len=self.pred_len,
                    period_len=self.period_len,
                    mode=router_mode,
                    hidden=getattr(configs, "triaxis_router_hidden", 16),
                    temperature=getattr(
                        configs, "triaxis_router_temperature", 1.0
                    ),
                )
            self.triaxis_expert_aux_weight = float(
                getattr(configs, "triaxis_expert_aux_weight", 0.2)
            )
            self.triaxis_route_aux_weight = float(
                getattr(configs, "triaxis_route_aux_weight", 0.1)
            )
            self.triaxis_oracle_temperature = float(
                getattr(configs, "triaxis_oracle_temperature", 0.2)
            )
            self.triaxis_route_target_granularity = getattr(
                configs, "triaxis_route_target_granularity", "point"
            )
            self.triaxis_expert_outputs = None
            self.triaxis_weights = None

        if self.use_safe_triaxis:
            if not self.use_weak_period_residual or not self.use_rcrf_fusion:
                raise ValueError(
                    "Safe-Regret TriAxis requires the complete A1 RCRF+NLinear anchor"
                )
            if self.use_periodic_residual_pe or self.use_dual_reliability_fusion:
                raise ValueError("Safe-Regret TriAxis requires the plain NLinear A1 anchor")
            cycle_period = getattr(configs, "safe_triaxis_cycle_period_len", 24)
            if cycle_period != self.period_len:
                raise ValueError("Safe-Regret cycle and phase periods must match")
            self.safe_triaxis_cycle_expert = InterCyclePatchResidualHead(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                period_len=cycle_period,
                d_model=getattr(configs, "safe_triaxis_cycle_d_model", 32),
                num_heads=getattr(configs, "safe_triaxis_cycle_heads", 4),
                ffn_dim=getattr(configs, "safe_triaxis_cycle_ffn_dim", 64),
                encoder_layers=getattr(
                    configs, "safe_triaxis_cycle_encoder_layers", 1
                ),
                decoder_layers=getattr(
                    configs, "safe_triaxis_cycle_decoder_layers", 1
                ),
                pe_type="none",
                relative_buckets=16,
                lff_frequencies=16,
                use_last_cycle_anchor=True,
                use_attention=True,
                label_len=getattr(configs, "label_len", 0),
                dropout=0.0,
                prediction_head="decoder",
                anchor_mode="last_cycle",
            )
            self.safe_triaxis_router = SafeRegretTriAxisRouter(
                pred_len=self.pred_len,
                period_len=self.period_len,
                hidden=getattr(configs, "safe_triaxis_router_hidden", 16),
                origins=getattr(configs, "safe_triaxis_rolling_origins", 4),
                trajectory_window_cycles=getattr(
                    configs, "safe_triaxis_trajectory_window_cycles", 4
                ),
                recency_decay=getattr(
                    configs, "safe_triaxis_rolling_recency_decay", 0.5
                ),
                correction_clip=getattr(
                    configs, "safe_triaxis_correction_clip", 2.0
                ),
                max_accept=getattr(configs, "safe_triaxis_max_accept", 1.0),
                use_horizon_prior=getattr(
                    configs, "safe_triaxis_use_horizon_prior", False
                ),
                horizon_prior_init=getattr(
                    configs, "safe_triaxis_horizon_prior_init", 0.05
                ),
            )
            self.safe_triaxis_route_aux_weight = float(
                getattr(configs, "safe_triaxis_route_aux_weight", 0.0)
            )
            self.safe_triaxis_cycle_aux_weight = float(
                getattr(configs, "safe_triaxis_cycle_aux_weight", 0.0)
            )
            self.safe_triaxis_nonreg_weight = float(
                getattr(configs, "safe_triaxis_nonreg_weight", 0.0)
            )
            self.safe_triaxis_cvar_weight = float(
                getattr(configs, "safe_triaxis_cvar_weight", 0.0)
            )
            self.safe_triaxis_regret_margin = float(
                getattr(configs, "safe_triaxis_regret_margin", 0.02)
            )
            self.safe_triaxis_oracle_temperature = float(
                getattr(configs, "safe_triaxis_oracle_temperature", 0.1)
            )
            self.safe_triaxis_anchor_frozen = False
            self.safe_triaxis_anchor_output = None
            self.safe_triaxis_expert_outputs = None
            self.safe_triaxis_weights = None

        self.use_time_mark_adjustment = getattr(configs, "use_time_mark_adjustment", False)
        if self.use_time_mark_adjustment:
            self.time_mark_adjustment = TimeMarkAdjustmentHead(
                mark_dim=getattr(configs, "time_mark_dim", 5),
                enc_in=self.enc_in,
                hidden=getattr(configs, "time_mark_hidden", 32),
            )

        self.use_phase_local_trend = getattr(configs, "use_phase_local_trend", False)
        if not self.use_residual_head:
            self.use_phase_local_trend = False
        if self.use_phase_local_trend:
            self.phase_local_trend = PhaseLocalTrendHead(
                num_periods_output=self.num_periods_output,
                enc_in=self.enc_in,
                window=getattr(configs, "phase_local_trend_window", 3),
                gate_init=getattr(configs, "phase_local_trend_gate_init", 0.0),
            )

        self.use_phase_uncertainty_shrinkage = getattr(
            configs, "use_phase_uncertainty_shrinkage", False
        )
        if self.use_phase_uncertainty_shrinkage:
            self.phase_uncertainty_shrinkage = PhaseUncertaintyShrinkage(
                enc_in=self.enc_in,
                min_reliability=getattr(configs, "phase_uncertainty_min", 0.35),
                trend_gate_init=getattr(configs, "phase_uncertainty_trend_gate_init", 0.05),
            )

        self.use_phase_period_level_calibration = getattr(
            configs, "use_phase_period_level_calibration", False
        )
        if self.use_phase_period_level_calibration:
            self.phase_period_level_calibration = PhasePeriodLevelCalibration(
                num_periods_output=self.num_periods_output,
                enc_in=self.enc_in,
                slope_window=getattr(configs, "phase_level_slope_window", 3),
                level_gate_init=getattr(configs, "phase_level_calib_gate_init", 0.1),
                slope_gate_init=getattr(configs, "phase_level_slope_gate_init", 0.05),
            )

        self.use_phase_sparse_event_calibration = getattr(
            configs, "use_phase_sparse_event_calibration", False
        )
        if self.use_phase_sparse_event_calibration:
            self.phase_sparse_event_calibration = PhaseSparseEventCalibration(
                enc_in=self.enc_in,
                window=getattr(configs, "phase_sparse_event_window", 3),
                gate_init=getattr(configs, "phase_sparse_event_gate_init", 0.05),
                max_boost=getattr(configs, "phase_sparse_event_max_boost", 1.0),
                temperature=getattr(configs, "phase_sparse_event_temperature", 0.2),
            )

        self.use_phase_noise_hifreq_damping = getattr(
            configs, "use_phase_noise_hifreq_damping", False
        )
        if self.use_phase_noise_hifreq_damping:
            self.phase_noise_hifreq_damping = PhaseNoiseHighFreqDamping(
                strength=getattr(configs, "phase_noise_hifreq_strength", 0.5),
                noise_threshold=getattr(configs, "phase_noise_hifreq_threshold", 1.0),
                noise_temperature=getattr(configs, "phase_noise_hifreq_temperature", 0.2),
                window=getattr(configs, "phase_noise_hifreq_window", 7),
            )

        # loss configuration
        self.use_huber_loss = getattr(configs, "use_huber_loss", False)
        self.huber_delta = getattr(configs, "huber_delta", 1.0)

        # expose: embedding -> [CrossPhaseRouting] x N -> predictor (P_out)
        self.phase_layers = getattr(configs, "phase_layers", 1)

        # Top-level embedding: projects (B, C, L, P_in) -> (B, C, L, D)
        self.embedding = PhaseEmbedding(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim,
            hidden=self.phase_encoder_hidden,
            use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
            dropout=getattr(configs, "phase_encoder_dropout", 0.0),
        )

        # Routing layers: Cross-phase routing with optional linear in/out projections
        routing_units = []
        if self.phase_layers == 1:
            routing_units.append(
                CrossPhaseRoutingUnit(
                    apply_in_proj=False,
                    apply_out_proj=False,
                    num_periods_input=self.num_periods_input,
                    latent_dim=self.latent_dim,
                    phase_encoder_hidden=self.phase_encoder_hidden,
                    predictor_hidden=self.predictor_hidden,
                    phase_attn_heads=self.phase_attn_heads,
                    phase_attn_dropout=self.phase_attn_dropout,
                    phase_attn_use_relpos=self.phase_attn_use_relpos,
                    period_len=self.period_len,
                    phase_attn_window=self.phase_attn_window,
                    phase_attention_dim=self.phase_attention_dim,
                    phase_num_routers=self.phase_num_routers,
                    phase_use_pos_embed=self.phase_use_pos_embed,
                    phase_pos_dropout=self.phase_pos_dropout,
                    phase_use_circular_pos=self.phase_use_circular_pos,
                    phase_use_circular_attn_bias=self.phase_use_circular_attn_bias,
                    phase_circular_attn_bias_scale=self.phase_circular_attn_bias_scale,
                    phase_encoder_use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
                    phase_encoder_dropout=getattr(configs, "phase_encoder_dropout", 0.0),
                    predictor_use_mlp=getattr(configs, "predictor_use_mlp", False),
                    predictor_dropout=getattr(configs, "predictor_dropout", 0.0),
                )
            )
        else:
            for li in range(self.phase_layers):
                is_first = li == 0
                is_last = li == self.phase_layers - 1
                routing_units.append(
                    CrossPhaseRoutingUnit(
                        apply_in_proj=not is_first,
                        apply_out_proj=not is_last,
                        num_periods_input=self.num_periods_input,
                        latent_dim=self.latent_dim,
                        phase_encoder_hidden=self.phase_encoder_hidden,
                        predictor_hidden=self.predictor_hidden,
                        phase_attn_heads=self.phase_attn_heads,
                        phase_attn_dropout=self.phase_attn_dropout,
                        phase_attn_use_relpos=self.phase_attn_use_relpos,
                        period_len=self.period_len,
                        phase_attn_window=self.phase_attn_window,
                        phase_attention_dim=self.phase_attention_dim,
                        phase_num_routers=self.phase_num_routers,
                        phase_use_pos_embed=self.phase_use_pos_embed,
                        phase_pos_dropout=self.phase_pos_dropout,
                        phase_use_circular_pos=self.phase_use_circular_pos,
                        phase_encoder_use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
                        phase_encoder_dropout=getattr(configs, "phase_encoder_dropout", 0.0),
                        predictor_use_mlp=getattr(configs, "predictor_use_mlp", False),
                        predictor_dropout=getattr(configs, "predictor_dropout", 0.0),
                    )
                )
        self.routing_layers = nn.ModuleList(routing_units)

        # Top-level predictor to P_out: maps (B, C, L, D) -> (B, C, L, P_out)
        self.predictor = PhasePredictor(
            p_out=self.num_periods_output,
            latent_dim=self.latent_dim,
            hidden=self.predictor_hidden,
            use_mlp=getattr(configs, "predictor_use_mlp", False),
            dropout=getattr(configs, "predictor_dropout", 0.0),
        )

        # Adaptive phase alignment. Constructed LAST so that toggling the flag
        # does not shift the RNG draws consumed by the shared modules above;
        # with the same seed, flag-on and flag-off share identical parameter
        # initialization for everything except phase_alignment itself.
        self.use_phase_align = getattr(configs, "use_phase_align", False)
        self.use_phase_warp = getattr(configs, "use_phase_warp", False)
        if self.use_phase_align and self.use_phase_warp:
            raise ValueError(
                "use_phase_align and use_phase_warp are mutually exclusive"
            )
        if self.use_phase_align:
            mark_dim = getattr(configs, "phase_align_mark_dim", None)
            if mark_dim is None:
                mark_dim = getattr(configs, "time_mark_dim", 4)
            self.phase_align_mark_dim = mark_dim
            self.phase_alignment = PhaseAlignment(
                mark_dim=mark_dim,
                hidden=getattr(configs, "phase_align_hidden", 8),
                use_position_encoding=getattr(
                    configs, "phase_align_position_encoding", False
                ),
                chunk_t=getattr(configs, "phase_align_chunk", 240),
            )
        if self.use_phase_warp:
            mark_dim = getattr(configs, "phase_warp_mark_dim", None)
            if mark_dim is None:
                mark_dim = getattr(configs, "time_mark_dim", 4)
            self.phase_warp_mark_dim = mark_dim
            self.phase_warp = PhaseWarping(
                mark_dim=mark_dim,
                hidden=getattr(configs, "phase_warp_hidden", 8),
                chunk_t=getattr(configs, "phase_warp_chunk", 240),
            )
        # Phase-conditioned amplitude calibration builds on the (warped) phase
        # representation, so it is constructed after phase_align/phase_warp to
        # keep the flag-off initialization identical for those modules too.
        self.use_phase_amp_calib = getattr(configs, "use_phase_amp_calib", False)
        if self.use_phase_amp_calib:
            self.phase_amp_calib = PhaseAmpCalibration(
                hidden=getattr(configs, "phase_amp_calib_hidden", 8),
                max_scale=getattr(configs, "phase_amp_calib_max_scale", 2.0),
            )

        # Reliability-aware Adaptive Phase Evolution (RAPE): phase warp +
        # amplitude calibration + a per-sample reliability gate that fuses the
        # adapted representation with the original fixed-grid phase prior.
        # Constructed last; at construction warp and amp are identity, so the
        # fused output reduces to the identity phase for any gate value (warm
        # start), and flag-off keeps baseline initialization.
        self.use_phase_rape = getattr(configs, "use_phase_rape", False)
        if self.use_phase_rape:
            if self.use_phase_align or self.use_phase_warp or self.use_phase_amp_calib:
                raise ValueError(
                    "use_phase_rape is mutually exclusive with use_phase_align, "
                    "use_phase_warp and use_phase_amp_calib"
                )
            mark_dim = getattr(configs, "phase_rape_mark_dim", None)
            if mark_dim is None:
                mark_dim = getattr(configs, "time_mark_dim", 4)
            self.phase_rape_mark_dim = mark_dim
            self.phase_warp_mark_dim = mark_dim
            self.phase_warp = PhaseWarping(
                mark_dim=mark_dim,
                hidden=getattr(configs, "phase_warp_hidden", 8),
                chunk_t=getattr(configs, "phase_warp_chunk", 240),
            )
            self.phase_amp_calib = PhaseAmpCalibration(
                hidden=getattr(configs, "phase_amp_calib_hidden", 8),
                max_scale=getattr(configs, "phase_amp_calib_max_scale", 2.0),
            )
            self.reliability_gate = ReliabilityGate(
                hidden=getattr(configs, "phase_rape_gate_hidden", 8),
            )

        # Dynamic-phase mechanisms (experiment plan stages 2-5). Constructed
        # after every shared module (and RAPE) so that toggling any of these
        # flags does not shift the RNG draws consumed by the shared path: with
        # the same seed, flag-on and flag-off share identical parameter
        # initialization for everything except the new module itself. Each new
        # module is a warm-start identity: correction (delta=0 -> no shift),
        # rotation (theta=0 -> no rotation) and harmonic modulation
        # (gamma=1, beta=0 -> identity).
        self.use_phase_correction = getattr(configs, "use_phase_correction", False)
        self.phase_correction_hidden = getattr(
            configs, "phase_correction_hidden", self.latent_dim
        )
        if self.use_phase_correction:
            self.phase_correction = PhaseCorrection(
                dim=self.latent_dim,
                hidden=self.phase_correction_hidden,
            )

        self.use_phase_rotation = getattr(configs, "use_phase_rotation", False)
        self.phase_rotation_hidden = getattr(configs, "phase_rotation_hidden", 8)
        if self.use_phase_rotation:
            self.phase_rotation = PhaseRotation(
                cond_dim=self.num_periods_input,
                hidden=self.phase_rotation_hidden,
            )

        self.use_harmonic_modulation = getattr(
            configs, "use_harmonic_modulation", False
        )
        self.harmonic_modulation_hidden = getattr(
            configs, "harmonic_modulation_hidden", 8
        )
        self.harmonic_modulation_max_scale = getattr(
            configs, "harmonic_modulation_max_scale", 2.0
        )
        if self.use_harmonic_modulation:
            self.harmonic_modulation = HarmonicModulation(
                cond_dim=self.num_periods_input,
                hidden=self.harmonic_modulation_hidden,
                max_scale=self.harmonic_modulation_max_scale,
            )

        # Next-stage dynamic-phase mechanisms (paper plan stages 1 and 3).
        # Constructed after every shared module (and the stage 2-5 mechanisms
        # above) for the same RNG-draw preservation reason: flag-on and flag-off
        # share identical initialization for everything except the new module.
        # PhaseVelocity is a warm-start identity (velocity=0 -> no shift);
        # AdaptiveResidualGate starts at alpha = gate_init (default 0.5).
        self.use_phase_velocity = getattr(configs, "use_phase_velocity", False)
        self.phase_velocity_hidden = getattr(configs, "phase_velocity_hidden", 8)
        self.phase_velocity_scale = getattr(configs, "phase_velocity_scale", 0.1)
        if self.use_phase_velocity:
            self.phase_velocity = PhaseVelocity(
                dim=self.latent_dim,
                hidden=self.phase_velocity_hidden,
                velocity_scale=self.phase_velocity_scale,
            )

        self.use_adaptive_residual_gate = getattr(
            configs, "use_adaptive_residual_gate", False
        )
        self.adaptive_residual_gate_hidden = getattr(
            configs, "adaptive_residual_gate_hidden", 8
        )
        self.adaptive_residual_gate_init = getattr(
            configs, "adaptive_residual_gate_init", 0.5
        )
        if self.use_adaptive_residual_gate:
            self.adaptive_residual_gate = AdaptiveResidualGate(
                phase_dim=self.latent_dim,
                enc_in=self.enc_in,
                hidden=self.adaptive_residual_gate_hidden,
                gate_init=self.adaptive_residual_gate_init,
            )

        # Pure-phase mechanisms (pure-phase plan stages 1-4). Constructed after
        # every shared module (and the residual-gate mechanism above) so that
        # toggling any of these flags does not shift the RNG draws consumed by
        # the shared path: flag-on and flag-off share identical initialization
        # for everything except the new module itself. Each new module is a
        # warm-start identity: MultiScalePhase (zeta=0), PhaseDeformation
        # (rate=0 -> identity scatter), PhaseGraph (message=0). TrajectoryDecoder
        # is an alternative top-level predictor, not an identity.
        self.use_multiscale_phase = getattr(configs, "use_multiscale_phase", False)
        self.phase_multiscale_long_period = getattr(
            configs, "phase_multiscale_long_period", 2 * self.period_len
        )
        self.phase_multiscale_coarse = getattr(configs, "phase_multiscale_coarse", 2)
        if self.use_multiscale_phase:
            if self.phase_multiscale_long_period != self.period_len * self.phase_multiscale_coarse:
                raise ValueError(
                    "phase_multiscale_long_period must equal period_len * phase_multiscale_coarse "
                    "so the coarse long-period view stays phase-aligned"
                )
            self.multiscale_phase = MultiScalePhase(
                latent_dim=self.latent_dim,
                period_len=self.period_len,
                num_periods_input=self.num_periods_input,
                coarse=self.phase_multiscale_coarse,
            )

        self.use_phase_deformation = getattr(configs, "use_phase_deformation", False)
        self.phase_deformation_hidden = getattr(
            configs, "phase_deformation_hidden", 8
        )
        self.phase_deformation_scale = getattr(
            configs, "phase_deformation_scale", 0.2
        )
        if self.use_phase_deformation:
            self.phase_deformation = PhaseDeformation(
                dim=self.latent_dim,
                hidden=self.phase_deformation_hidden,
                velocity_scale=self.phase_deformation_scale,
            )

        self.use_phase_graph = getattr(configs, "use_phase_graph", False)
        self.phase_graph_hidden = getattr(configs, "phase_graph_hidden", 16)
        self.phase_graph_k = getattr(configs, "phase_graph_k", 2)
        if self.use_phase_graph:
            self.phase_graph = PhaseGraph(
                dim=self.latent_dim,
                hidden=self.phase_graph_hidden,
                k=self.phase_graph_k,
            )

        self.use_trajectory_decoder = getattr(
            configs, "use_trajectory_decoder", False
        )
        self.phase_decoder_hidden = getattr(configs, "phase_decoder_hidden", 64)
        self.phase_decoder_order = getattr(configs, "phase_decoder_order", 2)
        if self.use_trajectory_decoder:
            self.trajectory_decoder = TrajectoryDecoder(
                latent_dim=self.latent_dim,
                p_out=self.num_periods_output,
                hidden=self.phase_decoder_hidden,
                order=self.phase_decoder_order,
            )

        # Residual-topology experiment.  These modules are constructed after
        # every shared component so enabling them cannot change shared-module
        # initialization.  Their projections are zero-initialized, making the
        # additive, long-latent, layer-wise and hybrid modes exact warm starts
        # of the original phase-only model.
        self.use_additive_output_residual = getattr(
            configs, "use_additive_output_residual", False
        )
        self.use_topology_output_convex_residual = getattr(
            configs, "use_topology_output_convex_residual", False
        )
        self.use_latent_long_residual = getattr(
            configs, "use_latent_long_residual", False
        )
        self.use_layerwise_latent_residual = getattr(
            configs, "use_layerwise_latent_residual", False
        )
        self.use_layerwise_output_convex = getattr(
            configs, "use_layerwise_output_convex", False
        )
        self.use_layerwise_output_additive = getattr(
            configs, "use_layerwise_output_additive", False
        )
        if not self.use_residual_head:
            self.use_additive_output_residual = False
            self.use_topology_output_convex_residual = False
            self.use_latent_long_residual = False
            self.use_layerwise_latent_residual = False
            self.use_layerwise_output_convex = False
            self.use_layerwise_output_additive = False
        if self.use_latent_long_residual and self.use_layerwise_latent_residual:
            raise ValueError(
                "use_latent_long_residual and use_layerwise_latent_residual "
                "are mutually exclusive"
            )
        if self.use_layerwise_output_convex and self.use_layerwise_output_additive:
            raise ValueError(
                "use_layerwise_output_convex and use_layerwise_output_additive "
                "are mutually exclusive"
            )
        if self.use_additive_output_residual:
            self.additive_output_residual = AdditiveOutputResidualHead(
                self.seq_len, self.pred_len
            )
            gate_init = float(
                getattr(configs, "additive_output_residual_gate_init", 0.5)
            )
            gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
            gate_logit = torch.logit(torch.tensor(gate_init))
            self.additive_output_residual_gate = nn.Parameter(
                torch.full((1, 1, self.enc_in), float(gate_logit))
            )
        if self.use_topology_output_convex_residual:
            # Semantically identical to the existing shared NLinear residual
            # branch, but constructed here (after all shared modules) so the
            # R0/R1 comparison keeps shared initialization exactly matched.
            self.topology_output_convex_residual = WeakPeriodResidualHead(
                self.seq_len, self.pred_len
            )
            gate_init = float(
                getattr(configs, "topology_output_convex_gate_init", 0.5)
            )
            gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
            gate_logit = torch.logit(torch.tensor(gate_init))
            self.topology_output_convex_gate = nn.Parameter(
                torch.full((1, 1, self.enc_in), float(gate_logit))
            )
        if self.use_latent_long_residual:
            self.latent_residual_path = LatentResidualPath(
                self.latent_dim, num_injections=1
            )
        elif self.use_layerwise_latent_residual:
            self.latent_residual_path = LatentResidualPath(
                self.latent_dim, num_injections=len(self.routing_layers)
            )

        # Layer-wise output residual (A1/A2): fuse an input-derived residual
        # onto every intermediate routing layer's phase-series prediction.  On
        # 1-layer models there are no intermediate layers, so these modes reduce
        # exactly to their single-point parents (R1 convex / R2 additive), which
        # the presets enable via use_topology_output_convex_residual /
        # use_additive_output_residual.  Gates are per-channel (1, C, 1, 1) so
        # they broadcast over the (B, C, num_slots, P) phase-series tensor.
        num_intermediate = max(self.phase_layers - 1, 0)
        self.layerwise_convex_residual = None
        self.layerwise_convex_gates = None
        self.layerwise_additive_residual = None
        self.layerwise_additive_gates = None
        if num_intermediate > 0:
            if self.use_layerwise_output_convex:
                self.layerwise_convex_residual = nn.ModuleList(
                    PhaseSlotResidualHead(
                        self.seq_len,
                        self.num_periods_input,
                        self.period_len,
                        anchor=True,
                    )
                    for _ in range(num_intermediate)
                )
                gate_init = float(
                    getattr(configs, "layerwise_output_convex_gate_init", 0.0)
                )
                gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
                gate_logit = torch.logit(torch.tensor(gate_init))
                self.layerwise_convex_gates = nn.ParameterList(
                    nn.Parameter(torch.full((1, self.enc_in, 1, 1), float(gate_logit)))
                    for _ in range(num_intermediate)
                )
            if self.use_layerwise_output_additive:
                self.layerwise_additive_residual = nn.ModuleList(
                    PhaseSlotResidualHead(
                        self.seq_len,
                        self.num_periods_input,
                        self.period_len,
                        anchor=False,
                    )
                    for _ in range(num_intermediate)
                )
                gate_init = float(
                    getattr(configs, "layerwise_output_additive_gate_init", 0.5)
                )
                gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
                gate_logit = torch.logit(torch.tensor(gate_init))
                self.layerwise_additive_gates = nn.ParameterList(
                    nn.Parameter(torch.full((1, self.enc_in, 1, 1), float(gate_logit)))
                    for _ in range(num_intermediate)
                )

    def freeze_safe_triaxis_anchor(self):
        """Freeze every loaded A1 parameter and leave only safe modules trainable."""
        if not self.use_safe_triaxis:
            raise RuntimeError("freeze_safe_triaxis_anchor requires use_safe_triaxis")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("safe_triaxis_"))
        self.safe_triaxis_anchor_frozen = True

    def on_train_epoch_start(self):
        # A1 is a pretrained anchor, not a jointly fine-tuned fourth expert.
        # Keep its dropout modules in evaluation mode while the newly added
        # cycle expert and router remain trainable.
        if getattr(self, "safe_triaxis_anchor_frozen", False):
            for name, module in self.named_children():
                if not name.startswith("safe_triaxis_"):
                    module.eval()
            self.safe_triaxis_cycle_expert.train()
            self.safe_triaxis_router.train()

    # phase rearrangement helpers
    @staticmethod
    def _to_phase_series(x_periods):
        """(B, C, P_in, L) -> (B, C, L, P_in)"""
        return x_periods.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _from_phase_steps_to_periods(y_phase_steps):
        """(B, C, L, P_out) -> (B, C, P_out, L)"""
        return y_phase_steps.permute(0, 1, 3, 2).contiguous()

    # forward pass
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        Input:  x_enc (B, seq_len, C)
        Output: y_hat (B, pred_len, C)
        Also returns intermediate Z (B,C,L,D) and future phase values (B,C,L,P_out) for analysis.
        """
        # Per-forward diagnostics used by frozen input-intervention attribution.
        # They are populated only for the plain RCRF path and detached at exit.
        self.last_phase_forecast = None
        self.last_residual_forecast = None
        self.last_rcrf_reliability = None
        self.last_rcrf_alpha = None

        # 1) RevIN normalization
        if self.use_revin:
            # RevIN expects (B, C, L)
            x_in, stats = self.revin.normalize(x_enc)
        else:
            x_in = x_enc.float()
        # 2) Use original input (no cross-channel fusion)
        x_fused = x_in  # (B, L, C)

        # 3) Ring padding to full periods
        x = x_fused.permute(0, 2, 1)  # (B, C, L_total)
        B, C, L = x.shape
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode="circular")  # (B, C, total_len_in)

        # 4) Split to periods (B, C, P_in, L)
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)

        # 5) Parallel by phase view (B, C, L, P_in)
        if self.use_phase_align or self.use_phase_warp or self.use_phase_rape:
            mark_dim = (
                self.phase_align_mark_dim if self.use_phase_align
                else self.phase_warp_mark_dim
            )
            if x_mark_enc is not None:
                mark = x_mark_enc.float()  # training passes float64; cast to float32
            else:
                mark = torch.zeros(
                    B, self.seq_len, mark_dim,
                    dtype=x.dtype, device=x.device,
                )
            if self.pad_seq_len > 0:
                mark = F.pad(mark, (0, 0, 0, self.pad_seq_len), mode="circular")
            if self.use_phase_align:
                phase_series = self.phase_alignment(x_periods, mark)
            elif self.use_phase_rape:
                phase_identity = self._to_phase_series(x_periods)
                phase_warped = self.phase_warp(x_periods, mark)
                phase_adapted = self.phase_amp_calib(phase_warped)
                gate = self.reliability_gate(x_in, phase_adapted, phase_identity)
                phase_series = (
                    gate.unsqueeze(-1).unsqueeze(-1) * phase_adapted
                    + (1.0 - gate).unsqueeze(-1).unsqueeze(-1) * phase_identity
                )
            else:
                phase_series = self.phase_warp(x_periods, mark)
        else:
            phase_series = self._to_phase_series(x_periods)
        # RCRF computes its reliability from the RAW phase series, before the
        # uncertainty shrinkage (and amp calibration) mutate the phase history,
        # so the correction modules never change the evidence that gates their
        # own contribution.
        phase_series_raw = phase_series
        if self.use_phase_uncertainty_shrinkage:
            phase_series = self.phase_uncertainty_shrinkage(phase_series)
        if self.use_phase_amp_calib:
            phase_series = self.phase_amp_calib(phase_series)

        # 6-8) Embedding -> routing layers -> top predictor
        # Initial latent from embedding.
        Z = self.embedding(phase_series)  # (B, C, L, D)

        # Multi-scale phase representation (pure-phase plan, stage 1): add a
        # gated long-period phase view at the same slot grid. zeta=0 at init
        # keeps the exact single-phase baseline.
        if self.use_multiscale_phase:
            Z = Z + self.multiscale_phase(phase_series)

        # Dynamic phase evolution: re-align the latent phase tokens along the
        # phase-slot axis. PhaseCorrection (static per-slot offset) and
        # PhaseVelocity (cumulative drift) precede the nonlinear deformation
        # field; the three are mutually exclusive, with deformation the most
        # expressive (rate + stretch -> non-uniform warp).
        if self.use_phase_correction:
            Z = self.phase_correction(Z)
        elif self.use_phase_velocity:
            Z = self.phase_velocity(Z)
        elif self.use_phase_deformation:
            Z = self.phase_deformation(Z)

        # Phase rotation: rotate pairs of latent features by a predicted angle
        # conditioned on the input periodic features (stage 4).
        if self.use_phase_rotation:
            Z = self.phase_rotation(Z, phase_series)

        # Geometry-aware phase interaction (pure-phase plan, stage 3): explicit
        # circular-graph message passing over the phase slots before routing.
        if self.use_phase_graph:
            Z = self.phase_graph(Z)

        residual_latent_anchor = Z
        phase_series_cur = phase_series

        for layer_index, unit in enumerate(self.routing_layers):
            Z, y_phase_steps_p_in = unit(phase_series_cur, Z)
            if self.use_layerwise_latent_residual:
                Z = Z + self.latent_residual_path(
                    residual_latent_anchor, layer_index
                )
            if layer_index < len(self.routing_layers) - 1:
                # intermediate layers must produce P_in for the next layer
                if (
                    self.use_layerwise_output_convex
                    and self.layerwise_convex_residual is not None
                ):
                    resid = self.layerwise_convex_residual[layer_index](
                        x_in, phase_series_cur[:, :, -1, :]
                    )
                    gate = torch.sigmoid(self.layerwise_convex_gates[layer_index])
                    y_phase_steps_p_in = (
                        (1.0 - gate) * y_phase_steps_p_in + gate * resid
                    )
                elif (
                    self.use_layerwise_output_additive
                    and self.layerwise_additive_residual is not None
                ):
                    corr = self.layerwise_additive_residual[layer_index](x_in)
                    gate = torch.sigmoid(self.layerwise_additive_gates[layer_index])
                    y_phase_steps_p_in = y_phase_steps_p_in + gate * corr
                phase_series_cur = y_phase_steps_p_in

        if self.use_latent_long_residual:
            Z = Z + self.latent_residual_path(residual_latent_anchor)

        # Harmonic feature modulation: rescale/shift the routed latent from input
        # periodic features, between routing and prediction (stage 5).
        if self.use_harmonic_modulation:
            Z = self.harmonic_modulation(Z, phase_series)

        # final predictor to produce P_out. TrajectoryDecoder (pure-phase plan,
        # stage 4) replaces the linear/MLP predictor with a per-slot polynomial
        # trajectory over the future cycles; both have the same (B, C, L, P_out)
        # output signature.
        if self.use_trajectory_decoder:
            y_phase_steps = self.trajectory_decoder(Z)  # (B, C, L, P_out)
        else:
            y_phase_steps = self.predictor(Z)  # (B, C, L, P_out)
        if self.use_phase_local_trend:
            phase_trend = self.phase_local_trend(phase_series)
            if self.training:
                y_phase_steps = y_phase_steps + phase_trend - phase_trend.detach()
            else:
                y_phase_steps = y_phase_steps + phase_trend
        if self.use_phase_period_level_calibration:
            y_phase_steps = self.phase_period_level_calibration(y_phase_steps, phase_series)
        if self.use_phase_sparse_event_calibration:
            y_phase_steps = self.phase_sparse_event_calibration(y_phase_steps, phase_series)

        # 9) Reassemble to sequence (B, pred_len, C)
        y_periods = self._from_phase_steps_to_periods(y_phase_steps)  # (B, C, P_out, L)
        y_full = y_periods.reshape(B, C, -1)[..., : self.pred_len]  # (B, C, pred_len)
        y_hat = y_full.permute(0, 2, 1)  # (B, pred_len, C)

        anchored_phase_hat = None
        anchored_trajectory_hat = None
        anchored_anchor_hat = None
        if self.use_anchored_phase_cycle_fusion:
            # Retain the calibrated phase expert.  The complete A2 anchor is
            # constructed by the ordinary downstream residual/RCRF path first.
            anchored_phase_hat = y_hat

        # Safe-Regret keeps the unblended phase forecast as an atomic expert,
        # while the complete downstream A1 path remains the immutable anchor.
        if self.use_safe_triaxis:
            safe_phase_hat = y_hat

        if self.use_triaxis_fusion:
            # High-frequency damping belongs to the phase expert.  Applying it
            # after fusion would silently alter the trajectory and cycle axes.
            phase_hat = y_hat
            if self.use_phase_noise_hifreq_damping:
                phase_hat = self.phase_noise_hifreq_damping(
                    phase_hat, phase_series
                )
            trajectory_hat = self.triaxis_trajectory_expert(x_in)
            cycle_hat = self.triaxis_cycle_expert(x_in)
            y_hat, self.triaxis_weights = self.triaxis_router(
                phase_hat, trajectory_hat, cycle_hat, x_in
            )
            triaxis_experts_normalized = (
                phase_hat, trajectory_hat, cycle_hat
            )

        if self.use_phase_cycle_fusion:
            # High-frequency damping is a phase-path calibration.  Apply it to
            # the phase component before composition so it cannot destroy the
            # composer's mean/shape identifiability after fusion.
            if self.use_phase_noise_hifreq_damping:
                y_hat = self.phase_noise_hifreq_damping(
                    y_hat, phase_series
                )
            y_hat = self.phase_cycle_fusion(
                y_hat, x_in, phase_series_raw
            )

        phase_forecast_normalized = y_hat
        residual_forecast_normalized = None
        if self.use_weak_period_residual:
            if self.use_dual_reliability_fusion:
                linear_hat, periodic_hat = self.weak_period_residual.forward_components(
                    x_in, x_mark_enc=x_mark_enc, x_mark_dec=x_mark_dec
                )
                y_hat, _ = self.dual_reliability_fusion(
                    y_hat, linear_hat, periodic_hat, phase_series_raw
                )
            else:
                if self.use_periodic_residual_pe or self.intercycle_head_requires_marks:
                    residual_hat = self.weak_period_residual(
                        x_in, x_mark_enc=x_mark_enc, x_mark_dec=x_mark_dec
                    )
                else:
                    residual_hat = self.weak_period_residual(x_in)
                residual_forecast_normalized = residual_hat
                if self.use_rcrf_fusion:
                    if self.use_anchored_phase_cycle_fusion:
                        anchored_trajectory_hat = residual_hat
                    y_hat, _ = self.rcrf_fusion(y_hat, residual_hat, phase_series_raw)
                    if self.use_safe_triaxis:
                        safe_trajectory_hat = residual_hat
                else:
                    if self.use_adaptive_weak_period_gate:
                        residual_gate = self.adaptive_weak_period_gate(x_in, phase_series)
                    elif self.use_adaptive_residual_gate:
                        residual_gate = self.adaptive_residual_gate(Z, x_in)
                    else:
                        residual_gate = torch.sigmoid(self.weak_period_residual_gate)
                    y_hat = (1.0 - residual_gate) * y_hat + residual_gate * residual_hat

        if self.use_topology_output_convex_residual:
            residual_hat = self.topology_output_convex_residual(x_in)
            gate = torch.sigmoid(self.topology_output_convex_gate)
            y_hat = (1.0 - gate) * y_hat + gate * residual_hat

        if self.use_additive_output_residual:
            correction = self.additive_output_residual(x_in)
            gate = torch.sigmoid(self.additive_output_residual_gate)
            y_hat = y_hat + gate * correction

        if self.use_time_mark_adjustment and x_mark_dec is not None:
            time_adjustment = self.time_mark_adjustment(
                x_mark_dec.float(), self.pred_len
            )
            y_hat = y_hat + time_adjustment

        if (
            self.use_phase_noise_hifreq_damping
            and not self.use_triaxis_fusion
            and not self.use_phase_cycle_fusion
        ):
            y_hat = self.phase_noise_hifreq_damping(y_hat, phase_series)
            if self.use_anchored_phase_cycle_fusion:
                anchored_phase_hat = self.phase_noise_hifreq_damping(
                    anchored_phase_hat, phase_series
                )
                if anchored_trajectory_hat is not None:
                    anchored_trajectory_hat = self.phase_noise_hifreq_damping(
                        anchored_trajectory_hat, phase_series
                    )

        if self.use_safe_triaxis:
            # `y_hat` is now the *complete* A1 prediction, including RCRF and
            # every frozen output calibration.  The safe extension is applied
            # only after that anchor has been fully constructed.
            safe_anchor_hat = y_hat
            if self.use_phase_noise_hifreq_damping:
                safe_phase_hat = self.phase_noise_hifreq_damping(
                    safe_phase_hat, phase_series
                )
            safe_cycle_hat = self.safe_triaxis_cycle_expert(x_in)
            y_hat, self.safe_triaxis_weights = self.safe_triaxis_router(
                safe_anchor_hat,
                safe_phase_hat,
                safe_trajectory_hat,
                safe_cycle_hat,
                x_in,
            )
            safe_experts_normalized = (
                safe_phase_hat, safe_trajectory_hat, safe_cycle_hat
            )

        if self.use_anchored_phase_cycle_fusion:
            # The incumbent remains a complete, trainable subgraph.  ICPT only
            # contributes bounded orthogonal innovations, all exactly zero at
            # initialization, after every A2 output calibration has run.
            anchored_anchor_hat = y_hat
            # In the strict single-stage setting the A2 anchor is optimized
            # exclusively through its own loss.  Detaching *all* A2-derived
            # composer inputs is necessary: detaching only the final additive
            # anchor leaves a gradient path through the correction features.
            # The composer still sees the same numerical values and remains a
            # fully trainable correction of the current A2 forecast.
            if self.anchored_pctf_anchorless:
                # Ablation 2: no A2 branch exists.  The composer corrects
                # PhaseFormer's own forecast in place; all three composer views
                # are the same signal and there is no trajectory predictor.
                composer_anchor = y_hat
                composer_phase = y_hat
                composer_trajectory = y_hat
                trajectory_predictor = None
            else:
                composer_anchor = anchored_anchor_hat
                composer_phase = anchored_phase_hat
                composer_trajectory = anchored_trajectory_hat
                trajectory_predictor = self.weak_period_residual
            composer_phase_series = phase_series_raw
            if self.anchored_pctf_detach_composer_inputs:
                composer_anchor = composer_anchor.detach()
                composer_phase = composer_phase.detach()
                composer_trajectory = composer_trajectory.detach()
                composer_phase_series = composer_phase_series.detach()
            y_hat = self.anchored_phase_cycle_fusion(
                composer_anchor,
                composer_phase,
                composer_trajectory,
                x_in,
                composer_phase_series,
                trajectory_predictor=trajectory_predictor,
            )

        # 10) De-normalization
        if self.use_revin:
            y_hat = self.revin.denormalize(y_hat, stats)
            if self.use_rcrf_fusion and residual_forecast_normalized is not None:
                self.last_phase_forecast = self.revin.denormalize(
                    phase_forecast_normalized, stats
                ).detach()
                self.last_residual_forecast = self.revin.denormalize(
                    residual_forecast_normalized, stats
                ).detach()
            if self.use_triaxis_fusion:
                self.triaxis_expert_outputs = tuple(
                    self.revin.denormalize(expert, stats)
                    for expert in triaxis_experts_normalized
                )
            if self.use_safe_triaxis:
                self.safe_triaxis_anchor_output = self.revin.denormalize(
                    safe_anchor_hat, stats
                )
                self.safe_triaxis_expert_outputs = tuple(
                    self.revin.denormalize(expert, stats)
                    for expert in safe_experts_normalized
                )
            if self.use_anchored_phase_cycle_fusion:
                self.anchored_pctf_anchor_output = self.revin.denormalize(
                    anchored_anchor_hat, stats
                )
                self.anchored_pctf_cycle_output = self.revin.denormalize(
                    self.anchored_phase_cycle_fusion.cycle_for_auxiliary, stats
                )
                # Corrections are additive, so de-normalization applies only
                # the per-sample scale and must not add the input mean.
                self.anchored_pctf_level_correction_output = (
                    self.anchored_phase_cycle_fusion
                    .level_correction_for_auxiliary * stats[1]
                )
                self.anchored_pctf_shape_correction_output = (
                    self.anchored_phase_cycle_fusion
                    .shape_correction_for_auxiliary * stats[1]
                )
        elif self.use_triaxis_fusion:
            self.triaxis_expert_outputs = triaxis_experts_normalized
        elif self.use_safe_triaxis:
            self.safe_triaxis_anchor_output = safe_anchor_hat
            self.safe_triaxis_expert_outputs = safe_experts_normalized
        elif self.use_anchored_phase_cycle_fusion:
            self.anchored_pctf_anchor_output = anchored_anchor_hat
            self.anchored_pctf_cycle_output = (
                self.anchored_phase_cycle_fusion.cycle_for_auxiliary
            )
            self.anchored_pctf_level_correction_output = (
                self.anchored_phase_cycle_fusion.level_correction_for_auxiliary
            )
            self.anchored_pctf_shape_correction_output = (
                self.anchored_phase_cycle_fusion.shape_correction_for_auxiliary
            )

        if self.use_rcrf_fusion and residual_forecast_normalized is not None:
            if not self.use_revin:
                self.last_phase_forecast = phase_forecast_normalized.detach()
                self.last_residual_forecast = residual_forecast_normalized.detach()
            self.last_rcrf_reliability = self.rcrf_fusion.last_r
            self.last_rcrf_alpha = self.rcrf_fusion.last_alpha

        return y_hat, Z, y_phase_steps

    # Lightning training steps
    def _compute_loss(self, outputs, target):
        """Loss computation with Huber support."""
        loss_func = str(getattr(self.args.training_args, "loss_func", "mse")).lower()
        # Old direct model configs may only expose use_huber_loss.
        if self.use_huber_loss:
            loss_func = "huber"
        criterion = self._get_criterion(loss_func)
        return criterion(outputs, target)

    def _cyclewise_mse(self, forecast, target):
        """Return per-sample/channel/future-cycle MSE as ``(B,C,Q)``."""
        parts = []
        for start in range(0, self.pred_len, self.period_len):
            end = min(start + self.period_len, self.pred_len)
            parts.append(
                (forecast[:, start:end, :] - target[:, start:end, :])
                .square()
                .mean(dim=1)
            )
        return torch.stack(parts, dim=-1)

    def _safe_triaxis_auxiliary_loss(self, outputs, target):
        """Supervise abstention by regret relative to the frozen A1 anchor."""
        anchor = self.safe_triaxis_anchor_output[:, -self.pred_len :, :]
        experts = tuple(
            expert[:, -self.pred_len :, :]
            for expert in self.safe_triaxis_expert_outputs
        )
        if self.target_var_index != -1:
            index = self.target_var_index
            anchor = anchor[:, :, index : index + 1]
            experts = tuple(x[:, :, index : index + 1] for x in experts)
            outputs = outputs[:, :, index : index + 1]

        anchor_mse = self._cyclewise_mse(anchor, target)
        expert_mse = torch.stack(
            [self._cyclewise_mse(expert, target) for expert in experts], dim=-1
        )
        relative_gain = (
            (anchor_mse.unsqueeze(-1) - expert_mse)
            / anchor_mse.unsqueeze(-1).clamp_min(1e-8)
        ).detach()
        noop_score = torch.zeros_like(relative_gain[..., :1])
        oracle_scores = torch.cat(
            (noop_score, relative_gain - self.safe_triaxis_regret_margin), dim=-1
        )
        oracle = torch.softmax(
            oracle_scores / self.safe_triaxis_oracle_temperature, dim=-1
        )
        logits = self.safe_triaxis_router.last_cycle_action_logits
        if self.target_var_index != -1:
            logits = logits[:, self.target_var_index : self.target_var_index + 1]
        action_ce = -(oracle * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        oracle_accept = oracle[..., 1:].sum(dim=-1)
        predicted_accept = self.safe_triaxis_router.last_cycle_accept.squeeze(-1)
        if self.target_var_index != -1:
            predicted_accept = predicted_accept[
                :, self.target_var_index : self.target_var_index + 1
            ]
        accept_calibration = F.mse_loss(predicted_accept, oracle_accept)
        route_loss = action_ce + accept_calibration

        cycle_aux = self._compute_loss(experts[2], target)
        candidate_mse = self._cyclewise_mse(outputs, target)
        relative_regret = (
            (candidate_mse - anchor_mse) / anchor_mse.clamp_min(1e-8)
        )
        positive_regret = F.relu(relative_regret)
        nonreg_loss = positive_regret.mean()
        flat = positive_regret.reshape(-1)
        tail_count = max(1, math.ceil(0.1 * flat.numel()))
        cvar_loss = flat.topk(tail_count).values.mean()
        auxiliary = (
            self.safe_triaxis_route_aux_weight * route_loss
            + self.safe_triaxis_cycle_aux_weight * cycle_aux
            + self.safe_triaxis_nonreg_weight * nonreg_loss
            + self.safe_triaxis_cvar_weight * cvar_loss
        )
        return auxiliary, {
            "route": route_loss,
            "cycle": cycle_aux,
            "nonreg": nonreg_loss,
            "cvar": cvar_loss,
        }

    def _anchored_pctf_auxiliary_loss(self, target):
        """Train ICPT in its admitted level/shape subspaces from step zero.

        The output correction is exactly zero initialized, so without this
        component supervision ICPT receives no main-loss gradient on the first
        optimizer step.  The auxiliary objective supervises only the internal
        cycle head; validation and checkpoint selection still use the fused
        forecast loss alone.
        """
        index = self.target_var_index
        if self.anchored_pctf_aux_target == "absolute":
            cycle = self.anchored_pctf_cycle_output[:, -self.pred_len :, :]
            if index != -1:
                cycle = cycle[:, :, index : index + 1]
            predicted_level, predicted_shape = (
                self.anchored_phase_cycle_fusion.decompose_forecast(cycle)
            )
            target_level, target_shape = (
                self.anchored_phase_cycle_fusion.decompose_forecast(target)
            )
        else:
            anchor = self.anchored_pctf_anchor_output[:, -self.pred_len :, :]
            predicted_level = self.anchored_pctf_level_correction_output
            predicted_shape = self.anchored_pctf_shape_correction_output
            if index != -1:
                anchor = anchor[:, :, index : index + 1]
                predicted_level = predicted_level[:, :, index : index + 1]
                predicted_shape = predicted_shape[:, :, index : index + 1]
            residual = target - anchor.detach()
            target_level, target_shape = (
                self.anchored_phase_cycle_fusion.decompose_residual_target(
                    residual
                )
            )
        shape_loss = self._compute_loss(predicted_shape, target_shape.detach())
        level_loss = self._compute_loss(predicted_level, target_level.detach())
        auxiliary = (
            self.anchored_pctf_shape_aux_weight * shape_loss
            + self.anchored_pctf_level_aux_weight * level_loss
        )
        gate_loss = auxiliary.new_zeros(())
        if self.anchored_pctf_gate_aux_weight > 0:
            level_correction = self.anchored_phase_cycle_fusion._cycles(
                predicted_level
            )
            shape_correction = self.anchored_phase_cycle_fusion._cycles(
                predicted_shape
            )
            level_target = self.anchored_phase_cycle_fusion._cycles(
                target_level
            )
            shape_target = self.anchored_phase_cycle_fusion._cycles(
                target_shape
            )

            def oracle_coefficient(correction, component_target):
                numerator = (
                    correction.detach() * component_target.detach()
                ).sum(dim=2)
                denominator = correction.detach().square().sum(
                    dim=2
                ).clamp_min(self.anchored_phase_cycle_fusion.eps)
                return (numerator / denominator).clamp(
                    -self.anchored_phase_cycle_fusion.correction_max,
                    self.anchored_phase_cycle_fusion.correction_max,
                ).permute(0, 2, 1)

            predicted_level_coefficient = (
                self.anchored_phase_cycle_fusion
                .level_coefficient_for_auxiliary
            )
            predicted_shape_coefficient = (
                self.anchored_phase_cycle_fusion
                .shape_coefficient_for_auxiliary
            )
            if index != -1:
                predicted_level_coefficient = predicted_level_coefficient[
                    :, index : index + 1
                ]
                predicted_shape_coefficient = predicted_shape_coefficient[
                    :, index : index + 1
                ]
            gate_loss = 0.5 * (
                F.smooth_l1_loss(
                    predicted_level_coefficient,
                    oracle_coefficient(level_correction, level_target),
                )
                + F.smooth_l1_loss(
                    predicted_shape_coefficient,
                    oracle_coefficient(shape_correction, shape_target),
                )
            )
            auxiliary = auxiliary + self.anchored_pctf_gate_aux_weight * gate_loss
        self.anchored_pctf_last_gate_aux_loss = gate_loss
        return auxiliary, shape_loss, level_loss

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        dec_inp = self._build_decoder_input(batch_y)

        outputs, Z, _ = self(
            x_enc=batch_x, x_mark_enc=batch_x_mark, x_dec=dec_inp, x_mark_dec=batch_y_mark
        )

        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        if (
            self.use_anchored_phase_cycle_fusion
            and self.anchored_pctf_decouple_anchor_gradient
        ):
            # Numerically this remains the same fused forecast.  Autograd,
            # however, sends the fused objective only through the ICPT/fusion
            # correction.  The incumbent A2 subgraph is trained exclusively by
            # its matched anchor objective below, avoiding reciprocal gradient
            # interference while retaining a single end-to-end fit call.
            anchor_for_gradient = self.anchored_pctf_anchor_output[
                :, -self.pred_len :, :
            ]
            outputs = (
                outputs - anchor_for_gradient
                + anchor_for_gradient.detach()
            )

        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        loss = self._compute_loss(outputs, target)
        fused_loss = loss

        if self.use_triaxis_fusion:
            experts = self.triaxis_expert_outputs
            weights = self.triaxis_weights
            if self.target_var_index != -1:
                index = self.target_var_index
                experts = tuple(x[:, :, index : index + 1] for x in experts)
                weights = weights[:, :, index : index + 1, :]
            expert_loss = torch.stack(
                [self._compute_loss(x[:, -self.pred_len :, :], target) for x in experts]
            ).mean()
            actual_errors = torch.stack(
                [
                    (x[:, -self.pred_len :, :] - target).abs()
                    for x in experts
                ],
                dim=-1,
            ).detach()
            if self.triaxis_route_target_granularity == "cycle":
                # The deployable router is factorized by future cycle.  Smooth
                # the supervision to the same granularity instead of asking it
                # to imitate a noisy pointwise winner sequence.
                oracle_parts = []
                for start in range(0, self.pred_len, self.period_len):
                    end = min(start + self.period_len, self.pred_len)
                    cycle_mse = actual_errors[:, start:end].square().mean(
                        dim=1, keepdim=True
                    )
                    cycle_oracle = torch.softmax(
                        -cycle_mse / self.triaxis_oracle_temperature, dim=-1
                    )
                    oracle_parts.append(cycle_oracle.expand(-1, end - start, -1, -1))
                oracle = torch.cat(oracle_parts, dim=1)
            else:
                oracle = torch.softmax(
                    -actual_errors / self.triaxis_oracle_temperature, dim=-1
                )
            route_loss = (
                oracle
                * (
                    oracle.clamp_min(1e-8).log()
                    - weights.clamp_min(1e-8).log()
                )
            ).sum(dim=-1).mean()
            loss = (
                loss
                + self.triaxis_expert_aux_weight * expert_loss
                + self.triaxis_route_aux_weight * route_loss
            )
            self.log("train_fused_loss", fused_loss, on_epoch=True)
            self.log("train_expert_loss", expert_loss, on_epoch=True)
            self.log("train_route_loss", route_loss, on_epoch=True)

        if self.use_safe_triaxis:
            safe_aux, safe_parts = self._safe_triaxis_auxiliary_loss(
                outputs, target
            )
            loss = loss + safe_aux
            self.log("train_fused_loss", fused_loss, on_epoch=True)
            self.log("train_safe_route_loss", safe_parts["route"], on_epoch=True)
            self.log("train_safe_cycle_loss", safe_parts["cycle"], on_epoch=True)
            self.log("train_safe_nonreg_loss", safe_parts["nonreg"], on_epoch=True)
            self.log("train_safe_cvar_loss", safe_parts["cvar"], on_epoch=True)
            self.log(
                "train_safe_global_accept",
                self.safe_triaxis_router.last_global_accept,
                on_epoch=True,
            )

        if self.use_anchored_phase_cycle_fusion:
            anchored_aux, shape_aux, level_aux = (
                self._anchored_pctf_auxiliary_loss(target)
            )
            loss = loss + anchored_aux
            anchor = self.anchored_pctf_anchor_output[:, -self.pred_len :, :]
            if self.target_var_index != -1:
                index = self.target_var_index
                anchor = anchor[:, :, index : index + 1]
            anchor_loss = self._compute_loss(anchor, target)
            loss = loss + self.anchored_pctf_anchor_loss_weight * anchor_loss
            self.log("train_fused_loss", fused_loss, on_epoch=True)
            self.log("train_pctf_shape_aux_loss", shape_aux, on_epoch=True)
            self.log("train_pctf_level_aux_loss", level_aux, on_epoch=True)
            self.log("train_pctf_anchor_loss", anchor_loss, on_epoch=True)
            self.log(
                "train_pctf_gate_aux_loss",
                self.anchored_pctf_last_gate_aux_loss,
                on_epoch=True,
            )

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def _set_anchored_pctf_correction_epoch(self, epoch):
        if not self.use_anchored_phase_cycle_fusion:
            return None
        epochs = self.anchored_pctf_correction_warmup_epochs
        if not epochs:
            return None
        if epochs == 1:
            scale = 1.0
        else:
            scale = min(float(epoch) / float(epochs - 1), 1.0)
        self.anchored_phase_cycle_fusion.set_correction_scale(scale)
        return scale

    def on_train_epoch_start(self):
        """Advance the optional single-stage ICPT correction curriculum."""
        scale = self._set_anchored_pctf_correction_epoch(self.current_epoch)
        if scale is None:
            return
        self.log("train_pctf_correction_scale", scale, on_epoch=True)

    def freeze_anchored_pctf_anchor(self):
        """Freeze every parameter except the single ICPT/fusion composer.

        This is a causal diagnostic for anchor drift, not a multi-checkpoint
        inference path.  The resulting checkpoint still contains one model.
        """
        if not self.use_anchored_phase_cycle_fusion:
            raise RuntimeError("anchor freezing requires anchored PCTF")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith("anchored_phase_cycle_fusion.")
            )

    def configure_optimizers(self):
        if (
            not self.use_anchored_phase_cycle_fusion
            or (
                self.anchored_pctf_anchor_lr_scale == 1.0
                and self.anchored_pctf_composer_lr_scale == 1.0
            )
        ):
            return super().configure_optimizers()
        base_lr = float(self.args.training_args.learning_rate)
        composer = []
        anchor = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("anchored_phase_cycle_fusion."):
                composer.append(parameter)
            else:
                anchor.append(parameter)
        groups = [{
            "params": composer,
            "lr": base_lr * self.anchored_pctf_composer_lr_scale,
        }]
        if anchor:
            groups.append({
                "params": anchor,
                "lr": base_lr * self.anchored_pctf_anchor_lr_scale,
            })
        optimizer = torch.optim.Adam(groups, lr=base_lr)
        if self.args.training_args.lr_schedule_config.type == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.training_args.lr_schedule_config.tmax,
                eta_min=1e-8,
            )
            return [optimizer], [scheduler]
        return optimizer

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)

        outputs, _, _ = self(
            x_enc=batch_x, x_mark_enc=batch_x_mark, x_dec=dec_inp, x_mark_dec=batch_y_mark
        )

        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        loss = self._compute_loss(outputs, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)

        outputs, _, _ = self(
            x_enc=batch_x, x_mark_enc=batch_x_mark, x_dec=dec_inp, x_mark_dec=batch_y_mark
        )

        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        from src.utils.metrics import metric
        m = metric(outputs.detach(), target.detach())
        self.log_dict({f"test_{k}": v for k, v in m.items()}, on_epoch=True)
        return m
