import os
from copy import deepcopy

import config.base_config as config_module
from src.dataset.data_info import DATASET_INFO


DEFAULT_NORM_HYPERS = dict(revin_affine=False, revin_eps=1e-5)
DEFAULT_HORIZONS = [96, 192, 336, 720]
ABLATION_MODES = {
    "weak_residual",
    "adaptive_residual",
    "time_mark",
    "phase_trend",
    "phase_uncertainty",
    "phase_level",
    "phase_hifreq",
    "phase_sparse_event",
    "phase_all",
    "phase_align",
    "phase_warp",
    "phase_amp_calib",
    "phase_rape",
    "best_nonresidual",
    # Dynamic-phase mechanisms (weak-residual-phaseformer plan stages 1-5).
    "phase_correction",
    "dyn_corr",
    "dyn_geo",
    "dyn_corr_geo",
    "dyn_geo_rot",
    "dyn_corr_geo_rot",
    "dyn_stack",
    "residual_full",
    "no_residual",
    "dyn_full",
    # Next-stage paper plan mechanisms (Adaptive Phase-Residual Trajectory
    # Modeling): stages 1-3 (velocity, circular bias, adaptive residual gate).
    "phase_velocity",
    "phase_vel_geo",
    "residual_adaptive",
    "next_full",
    # Pure-phase plan mechanisms (Adaptive Phase Geometry Forecasting):
    # stage 1 multi-scale representation, stage 2 deformation, stage 3 phase
    # graph / circular bias, stage 4 trajectory decoder, final pure_full.
    "multiscale_phase",
    "phase_deformation",
    "phase_geo",
    "phase_graph",
    "predictor_mlp",
    "trajectory_decoder",
    "pure_full",
    # Residual-topology experiment: output, latent and hybrid long skips.
    "residual_output_convex",
    "residual_output_additive",
    "residual_latent_long",
    "residual_latent_layerwise",
    "residual_hybrid",
    # Layer-wise output residuals (A1/A2): convex / additive fusion applied at
    # every routing depth instead of only the final output.
    "residual_output_layerwise_convex",
    "residual_output_layerwise_additive",
    # Cross-dataset golden-combo mechanisms (gold_combo_stability_v1): shared
    # phase stack (uncertainty min 0.2 / period-level 0.2 / high-frequency
    # 0.8-0.5-w7) with four output fusion variants.  The residual gate prior
    # alpha_0 = 0.5 is shared by all four.
    "gold_combo_fixed",
    "gold_combo_adaptive",
    "gold_combo_reliability_s0",
    "gold_combo_reliability_s2",
    # Period-position-encoded NLinear residual candidates.  All inherit the
    # frozen gold_combo_reliability_s2 stack and differ only in PE type.
    "rcrf_pe_st",
    "rcrf_pe_cycle",
    "rcrf_pe_harmonic",
    "rcrf_pe_traffic",
    "rcrf_pe_time2vec",
    "rcrf_pe_lff",
    "rcrf_pe_calendar",
    # Inter-Cycle Patch Transformer residual candidates (ICPT plan).  A3/A4 are
    # simple cycle baselines; rcrf_icpt_* swap the NLinear head for the ICPT
    # transformer with the position encoding given by the suffix.  All inherit
    # the frozen gold_combo_reliability_s2 stack and the RCRF formula unchanged.
    "rcrf_repeat_last_cycle",
    "rcrf_cycle_net",
    "rcrf_icpt_none",
    "rcrf_icpt_sincos",
    "rcrf_icpt_learned_abs",
    "rcrf_icpt_time2vec",
    "rcrf_icpt_rope",
    "rcrf_icpt_relative",
    "rcrf_icpt_alibi",
    "rcrf_icpt_lff",
    "rcrf_icpt_sincos_relative",
    "rcrf_icpt_calendar",
    # PatchTST-style ordered full-horizon ICPT head and its PE variants.
    "rcrf_icpt_horizon_cycle_anchor",
    "rcrf_icpt_horizon_none",
    "rcrf_icpt_horizon_sincos",
    "rcrf_icpt_horizon_learned_abs",
    "rcrf_icpt_horizon_time2vec",
    "rcrf_icpt_horizon_rope",
    "rcrf_icpt_horizon_relative",
    "rcrf_icpt_horizon_alibi",
    "rcrf_icpt_horizon_lff",
    "rcrf_icpt_horizon_sincos_relative",
    "rcrf_icpt_horizon_calendar",
    # Stage D mechanism ablations of the frozen ICPT-best candidate.
    "icpt_only",
    "icpt_fixed_fusion",
    "icpt_patch16",
    "icpt_no_anchor",
    "icpt_no_attention",
}

PERIODIC_RESIDUAL_PE_MODES = {
    "rcrf_pe_st": "st_informer",
    "rcrf_pe_cycle": "cycle",
    "rcrf_pe_harmonic": "harmonic",
    "rcrf_pe_traffic": "traffic",
    "rcrf_pe_time2vec": "time2vec",
    "rcrf_pe_lff": "lff",
    "rcrf_pe_calendar": "calendar",
}

# ICPT position encodings.  P0 is the no-PE architecture baseline; P1-P8 are
# the pure index-PE candidates ranked together; P9 (calendar) is scored
# separately because it consumes real timestamps, not just cycle indices.
INTERCYCLE_PE_MODES = {
    "rcrf_icpt_none": "none",
    "rcrf_icpt_sincos": "sincos",
    "rcrf_icpt_learned_abs": "learned_abs",
    "rcrf_icpt_time2vec": "time2vec",
    "rcrf_icpt_rope": "rope",
    "rcrf_icpt_relative": "relative",
    "rcrf_icpt_alibi": "alibi",
    "rcrf_icpt_lff": "lff",
    "rcrf_icpt_sincos_relative": "sincos_relative",
    "rcrf_icpt_calendar": "calendar",
}

INTERCYCLE_HORIZON_PE_MODES = {
    "rcrf_icpt_horizon_none": "none",
    "rcrf_icpt_horizon_sincos": "sincos",
    "rcrf_icpt_horizon_learned_abs": "learned_abs",
    "rcrf_icpt_horizon_time2vec": "time2vec",
    "rcrf_icpt_horizon_rope": "rope",
    "rcrf_icpt_horizon_relative": "relative",
    "rcrf_icpt_horizon_alibi": "alibi",
    "rcrf_icpt_horizon_lff": "lff",
    "rcrf_icpt_horizon_sincos_relative": "sincos_relative",
    "rcrf_icpt_horizon_calendar": "calendar",
}

# Fixed ICPT hyperparameters shared by every rcrf_icpt_* candidate so that the
# only difference between modes is the position encoding.
ICPT_FIXED_HYPERPARAMS = {
    "weak_period_residual_head_type": "intercycle",
    "intercycle_period_len": 24,
    "intercycle_d_model": 32,
    "intercycle_heads": 4,
    "intercycle_ffn_dim": 64,
    "intercycle_encoder_layers": 1,
    "intercycle_decoder_layers": 1,
    "intercycle_relative_buckets": 16,
    "intercycle_lff_frequencies": 16,
    "intercycle_use_last_cycle_anchor": True,
    "intercycle_use_attention": True,
    "intercycle_dropout": 0.0,
}

ICPT_HORIZON_FIXED_HYPERPARAMS = {
    "weak_period_residual_head_type": "intercycle",
    "intercycle_period_len": 24,
    "intercycle_d_model": 24,
    "intercycle_heads": 4,
    "intercycle_ffn_dim": 48,
    "intercycle_encoder_layers": 1,
    "intercycle_decoder_layers": 0,
    "intercycle_relative_buckets": 16,
    "intercycle_lff_frequencies": 16,
    "intercycle_use_last_cycle_anchor": False,
    "intercycle_use_attention": True,
    "intercycle_dropout": 0.0,
    "intercycle_prediction_head": "flatten",
    "intercycle_anchor_mode": "last_value",
}


def get_frequency(dataset_name):
    if dataset_name == "Exchange":
        return "d"
    if dataset_name in ["ETTh1", "ETTh2"]:
        return "h"
    return "t"


def get_dataset_horizons(dataset_name):
    return list(DEFAULT_HORIZONS)


def get_base_hyperparams(dataset_name, horizon):
    if dataset_name == "Weather":
        if horizon == 96:
            return dict(
                layers=3,
                latent_dim=8,
                phase_encoder_hidden=32,
                predictor_hidden=64,
                phase_num_routers=8,
                learning_rate=0.001,
                phase_attn_heads=1,
            )
        return dict(
            layers=2,
            latent_dim=8,
            phase_encoder_hidden=32,
            predictor_hidden=64,
            phase_num_routers=8,
            learning_rate=0.001,
            phase_attn_heads=1,
        )
    if dataset_name == "Exchange":
        return dict(
            layers=2,
            latent_dim=8,
            phase_encoder_hidden=32,
            predictor_hidden=64,
            phase_num_routers=8,
            learning_rate=0.001,
            phase_attn_heads=1,
        )
    if dataset_name == "ETTh1":
        if horizon in [96, 192, 336]:
            return dict(
                layers=3,
                latent_dim=4,
                phase_encoder_hidden=16,
                predictor_hidden=32,
                phase_num_routers=8,
                learning_rate=0.001,
                phase_attn_heads=1,
            )
        return dict(
            layers=3,
            latent_dim=32,
            phase_encoder_hidden=128,
            predictor_hidden=256,
            phase_num_routers=16,
            learning_rate=0.00015,
            phase_attn_heads=2,
            train_epochs=70,
            patience=14,
            seed=2026,
            phase_attn_dropout=0.0,
            huber_delta=0.3,
        )
    if dataset_name == "ETTh2":
        if horizon in [96, 192, 336]:
            return dict(
                layers=1,
                latent_dim=8,
                phase_encoder_hidden=32,
                predictor_hidden=64,
                phase_num_routers=8,
                learning_rate=0.001,
                phase_attn_heads=1,
            )
        return dict(
            layers=1,
            latent_dim=4,
            phase_encoder_hidden=8,
            predictor_hidden=8,
            phase_num_routers=4,
            learning_rate=0.001,
            phase_attn_heads=1,
        )
    if dataset_name == "ETTm1":
        return dict(
            layers=1 if horizon == 336 else 2,
            latent_dim=8,
            phase_encoder_hidden=32,
            predictor_hidden=64,
            phase_num_routers=8,
            learning_rate=0.001,
            phase_attn_heads=1,
        )
    if dataset_name == "ETTm2":
        return dict(
            layers=2 if horizon == 96 else 1,
            latent_dim=8,
            phase_encoder_hidden=32,
            predictor_hidden=64,
            phase_num_routers=8,
            learning_rate=0.001,
            phase_attn_heads=1,
        )
    if dataset_name == "Electricity":
        if horizon == 96:
            return dict(
                layers=2,
                latent_dim=8,
                phase_encoder_hidden=32,
                predictor_hidden=64,
                phase_num_routers=8,
                learning_rate=0.002,
                phase_attn_heads=1,
            )
        if horizon in [192, 720]:
            return dict(
                layers=1,
                latent_dim=128,
                phase_encoder_hidden=16,
                predictor_hidden=32,
                phase_num_routers=4,
                learning_rate=0.001,
                phase_attn_heads=4,
            )
        return dict(
            layers=2,
            latent_dim=8,
            phase_encoder_hidden=32,
            predictor_hidden=64,
            phase_num_routers=8,
            learning_rate=0.001,
            phase_attn_heads=1,
        )
    if dataset_name == "Traffic":
        if horizon == 96:
            return dict(
                layers=2,
                latent_dim=32,
                phase_encoder_hidden=64,
                predictor_hidden=128,
                phase_num_routers=1,
                learning_rate=0.001,
                phase_attn_heads=8,
                phase_attn_window=12,
                phase_use_pos_embed=False,
                phase_attn_use_relpos=True,
            )
        return dict(
            layers=1,
            latent_dim=128,
            phase_encoder_hidden=16,
            predictor_hidden=32,
            phase_num_routers=4,
            learning_rate=0.001,
            phase_attn_heads=4,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


_LATEST_POLICY = {
    ("Exchange", None): { 'scheme_name': 'latest_exchange_residual_mae',
      'use_weak_period_residual': True,
      'weak_period_residual_gate_init': 0.999,
      'learning_rate': 0.00013,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("Weather", 96): { 'scheme_name': 'latest_weather96_phase_uncert_level_hifreq_sparse_event_mae',
      'period_len': 12,
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.35,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.1,
      'use_phase_noise_hifreq_damping': True,
      'phase_noise_hifreq_strength': 0.8,
      'phase_noise_hifreq_threshold': 0.5,
      'phase_noise_hifreq_window': 7,
      'use_phase_sparse_event_calibration': True,
      'phase_sparse_event_window': 3,
      'phase_sparse_event_gate_init': 0.1,
      'phase_sparse_event_max_boost': 1.0,
      'phase_sparse_event_temperature': 0.2,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False,
      'batch_size': 64},
    ("Electricity", 336): { 'scheme_name': 'latest_electricity336_adaptive_residual_mae',
      'use_weak_period_residual': True,
      'use_adaptive_weak_period_gate': True,
      'weak_period_residual_gate_init': 0.5,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False,
      'batch_size': 64},
    ("ETTh2", 96): { 'scheme_name': 'latest_etth2_adaptive_residual_mae',
      'use_weak_period_residual': True,
      'use_adaptive_weak_period_gate': True,
      'weak_period_residual_gate_init': 0.2,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTh2", 192): { 'scheme_name': 'latest_etth2_adaptive_residual_mae',
      'use_weak_period_residual': True,
      'use_adaptive_weak_period_gate': True,
      'weak_period_residual_gate_init': 0.2,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTh2", 336): { 'scheme_name': 'latest_etth2_adaptive_residual_mae',
      'use_weak_period_residual': True,
      'use_adaptive_weak_period_gate': True,
      'weak_period_residual_gate_init': 0.2,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False,
      'patience': 14},
    ("ETTh1", 96): { 'scheme_name': 'latest_etth1_phase_uncertainty_level_calib',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.35,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.1},
    ("ETTh1", 192): { 'scheme_name': 'latest_etth1_phase_uncertainty_light',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.6,
      'phase_uncertainty_trend_gate_init': 0.05},
    ("ETTh1", 336): { 'scheme_name': 'latest_etth1_phase_uncertainty_light',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.6,
      'phase_uncertainty_trend_gate_init': 0.05},
    ("ETTh1", 720): { 'scheme_name': 'latest_etth1_phase_uncertainty_level_calib_light',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.8,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.05,
      'learning_rate': 0.0001},
    ("ETTm1", 96): { 'scheme_name': 'latest_ettm1_phase_uncertainty_level_calib_hifreq_mae',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.35,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.1,
      'use_phase_noise_hifreq_damping': True,
      'phase_noise_hifreq_strength': 0.8,
      'phase_noise_hifreq_threshold': 0.5,
      'phase_noise_hifreq_window': 7,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTm1", 192): { 'scheme_name': 'latest_ettm1_phase_uncertainty_level_calib_hifreq_mae',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.35,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.1,
      'use_phase_noise_hifreq_damping': True,
      'phase_noise_hifreq_strength': 0.8,
      'phase_noise_hifreq_threshold': 0.5,
      'phase_noise_hifreq_window': 7,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTm1", 336): { 'scheme_name': 'latest_ettm1_phase_uncertainty_level_calib_hifreq_mae',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.35,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.1,
      'use_phase_noise_hifreq_damping': True,
      'phase_noise_hifreq_strength': 0.8,
      'phase_noise_hifreq_threshold': 0.5,
      'phase_noise_hifreq_window': 7,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTm1", 720): { 'scheme_name': 'latest_ettm1_phase_uncertainty_level_calib_hifreq_mae',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.35,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.1,
      'use_phase_noise_hifreq_damping': True,
      'phase_noise_hifreq_strength': 0.8,
      'phase_noise_hifreq_threshold': 0.5,
      'phase_noise_hifreq_window': 7,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTm2", 96): { 'scheme_name': 'latest_ettm2_phase_uncertainty_level_calib_hifreq_mae',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.2,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.2,
      'use_phase_noise_hifreq_damping': True,
      'phase_noise_hifreq_strength': 0.8,
      'phase_noise_hifreq_threshold': 0.5,
      'phase_noise_hifreq_window': 7,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTm2", 192): { 'scheme_name': 'latest_ettm2_phase_uncertainty_level_calib_hifreq_mae',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.2,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.2,
      'use_phase_noise_hifreq_damping': True,
      'phase_noise_hifreq_strength': 0.8,
      'phase_noise_hifreq_threshold': 0.5,
      'phase_noise_hifreq_window': 7,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTm2", 336): { 'scheme_name': 'latest_ettm2_phase_uncertainty_level_calib_hifreq_mae',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.2,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.2,
      'use_phase_noise_hifreq_damping': True,
      'phase_noise_hifreq_strength': 0.8,
      'phase_noise_hifreq_threshold': 0.5,
      'phase_noise_hifreq_window': 7,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTm2", 720): { 'scheme_name': 'latest_ettm2_phase_uncertainty_level_calib_mae',
      'use_phase_uncertainty_shrinkage': True,
      'phase_uncertainty_min': 0.6,
      'phase_uncertainty_trend_gate_init': 0.05,
      'use_phase_period_level_calibration': True,
      'phase_level_slope_window': 3,
      'phase_level_slope_gate_init': 0.05,
      'phase_level_calib_gate_init': 0.1,
      'learning_rate': 0.0003,
      'loss_func': 'mae',
      'use_huber_loss': False},
    ("ETTh2", 720): { 'scheme_name': 'latest_etth2_residual_long',
      'use_weak_period_residual': True,
      'weak_period_residual_gate_init': 0.999},
}


def get_latest_overrides(dataset_name, horizon):
    """Dataset-aware latest policy.

    Only enable weak-period mechanisms where full research evidence showed a
    positive signal; otherwise preserve the original phase-only design.

    The policy is a declarative table keyed by ``(dataset_name, horizon)`` where
    ``horizon is None`` applies to every horizon of that dataset. The returned
    dict is a deep copy so callers can mutate it safely.
    """
    if (dataset_name, horizon) in _LATEST_POLICY:
        return deepcopy(_LATEST_POLICY[(dataset_name, horizon)])
    fallback = _LATEST_POLICY.get((dataset_name, None))
    if fallback is not None:
        return deepcopy(fallback)
    return dict({'scheme_name': 'latest_original_guardrail'})


def get_ablation_overrides(mode):
    """Single-feature PhaseFormer ablations used by research scripts."""
    if mode == "weak_residual":
        return dict(
            scheme_name="weak_residual",
            use_weak_period_residual=True,
            weak_period_residual_gate_init=0.2,
            weak_period_residual_head_type="shared",
        )
    if mode == "adaptive_residual":
        return dict(
            scheme_name="adaptive_residual",
            use_weak_period_residual=True,
            use_adaptive_weak_period_gate=True,
            weak_period_residual_gate_init=0.2,
            weak_period_residual_head_type="shared",
        )
    if mode == "time_mark":
        return dict(scheme_name="time_mark", use_time_mark_adjustment=True)
    if mode == "phase_trend":
        return dict(
            scheme_name="phase_trend",
            use_phase_local_trend=True,
            phase_local_trend_window=3,
            phase_local_trend_gate_init=0.0,
        )
    if mode == "phase_uncertainty":
        return dict(
            scheme_name="phase_uncertainty",
            use_phase_uncertainty_shrinkage=True,
            phase_uncertainty_min=0.35,
            phase_uncertainty_trend_gate_init=0.05,
        )
    if mode == "phase_level":
        return dict(
            scheme_name="phase_level",
            use_phase_period_level_calibration=True,
            phase_level_slope_window=3,
            phase_level_slope_gate_init=0.05,
            phase_level_calib_gate_init=0.1,
        )
    if mode == "phase_hifreq":
        return dict(
            scheme_name="phase_hifreq",
            use_phase_noise_hifreq_damping=True,
            phase_noise_hifreq_strength=0.5,
            phase_noise_hifreq_threshold=1.0,
            phase_noise_hifreq_temperature=0.2,
            phase_noise_hifreq_window=7,
        )
    if mode == "phase_sparse_event":
        return dict(
            scheme_name="phase_sparse_event",
            use_phase_sparse_event_calibration=True,
            phase_sparse_event_window=3,
            phase_sparse_event_gate_init=0.05,
            phase_sparse_event_max_boost=1.0,
            phase_sparse_event_temperature=0.2,
        )
    if mode == "phase_all":
        return dict(
            scheme_name="phase_all",
            use_phase_local_trend=True,
            phase_local_trend_window=3,
            phase_local_trend_gate_init=0.0,
            use_phase_uncertainty_shrinkage=True,
            phase_uncertainty_min=0.35,
            phase_uncertainty_trend_gate_init=0.05,
            use_phase_period_level_calibration=True,
            phase_level_slope_window=3,
            phase_level_slope_gate_init=0.05,
            phase_level_calib_gate_init=0.1,
            use_phase_noise_hifreq_damping=True,
            phase_noise_hifreq_strength=0.5,
            phase_noise_hifreq_threshold=1.0,
            phase_noise_hifreq_temperature=0.2,
            phase_noise_hifreq_window=7,
            use_phase_sparse_event_calibration=True,
            phase_sparse_event_window=3,
            phase_sparse_event_gate_init=0.05,
            phase_sparse_event_max_boost=1.0,
            phase_sparse_event_temperature=0.2,
        )
    if mode == "phase_align":
        return dict(
            scheme_name="phase_align",
            use_phase_align=True,
            phase_align_hidden=8,
            phase_align_position_encoding=False,
        )
    if mode == "phase_warp":
        return dict(
            scheme_name="phase_warp",
            use_phase_warp=True,
            phase_warp_hidden=8,
        )
    if mode == "phase_amp_calib":
        return dict(
            scheme_name="phase_amp_calib",
            use_phase_warp=True,
            phase_warp_hidden=8,
            use_phase_amp_calib=True,
            phase_amp_calib_hidden=8,
            phase_amp_calib_max_scale=2.0,
        )
    if mode == "phase_rape":
        return dict(
            scheme_name="phase_rape",
            use_phase_rape=True,
            phase_warp_hidden=8,
            phase_amp_calib_hidden=8,
            phase_amp_calib_max_scale=2.0,
            phase_rape_gate_hidden=8,
        )
    # Dynamic-phase mechanisms (weak-residual-phaseformer plan stages 1-5).
    # Cumulative ladder mirrors the search runner's MECHANISMS so the same
    # configs are reachable through build_hyperparams for full-budget confirms.
    if mode in ("dyn_corr", "phase_correction"):
        return dict(scheme_name=mode, use_phase_correction=True)
    if mode in ("dyn_corr_geo", "dyn_geo"):
        return dict(
            scheme_name=mode,
            use_phase_correction=True,
            phase_use_circular_pos=True,
        )
    if mode in ("dyn_corr_geo_rot", "dyn_geo_rot"):
        return dict(
            scheme_name=mode,
            use_phase_correction=True,
            phase_use_circular_pos=True,
            use_phase_rotation=True,
            phase_rotation_hidden=8,
        )
    if mode == "dyn_stack":
        return dict(
            scheme_name="dyn_stack",
            use_phase_correction=True,
            phase_use_circular_pos=True,
            use_phase_rotation=True,
            phase_rotation_hidden=8,
            use_harmonic_modulation=True,
            harmonic_modulation_hidden=8,
            harmonic_modulation_max_scale=2.0,
        )
    if mode == "residual_full":
        return dict(
            scheme_name="residual_full",
            use_weak_period_residual=True,
            weak_period_residual_gate_init=0.5,
            use_phase_local_trend=True,
            phase_local_trend_window=3,
            phase_local_trend_gate_init=0.0,
        )
    if mode == "no_residual":
        return dict(scheme_name="no_residual", use_residual_head=False)
    if mode == "dyn_full":
        return dict(
            scheme_name="dyn_full",
            use_phase_correction=True,
            phase_use_circular_pos=True,
            use_phase_rotation=True,
            phase_rotation_hidden=8,
            use_harmonic_modulation=True,
            harmonic_modulation_hidden=8,
            harmonic_modulation_max_scale=2.0,
            use_weak_period_residual=True,
            weak_period_residual_gate_init=0.5,
            use_phase_local_trend=True,
            phase_local_trend_window=3,
            phase_local_trend_gate_init=0.0,
        )
    # Next-stage paper plan mechanisms (Adaptive Phase-Residual Trajectory
    # Modeling). phase_velocity = A2/B1 (stage 1), phase_vel_geo = B2 (stage 2
    # adds the circular attention bias), residual_adaptive = R2 (stage 3) and
    # next_full = C (velocity + circular bias + adaptive residual gate).
    if mode == "phase_velocity":
        return dict(
            scheme_name="phase_velocity",
            use_phase_velocity=True,
            phase_velocity_hidden=8,
            phase_velocity_scale=0.1,
        )
    if mode == "phase_vel_geo":
        return dict(
            scheme_name="phase_vel_geo",
            use_phase_velocity=True,
            phase_velocity_hidden=8,
            phase_velocity_scale=0.1,
            phase_use_circular_attn_bias=True,
            phase_circular_attn_bias_scale=1.0,
        )
    if mode == "residual_adaptive":
        return dict(
            scheme_name="residual_adaptive",
            use_weak_period_residual=True,
            weak_period_residual_gate_init=0.5,
            use_phase_local_trend=True,
            phase_local_trend_window=3,
            phase_local_trend_gate_init=0.0,
            use_adaptive_residual_gate=True,
            adaptive_residual_gate_hidden=8,
            adaptive_residual_gate_init=0.5,
        )
    if mode == "next_full":
        return dict(
            scheme_name="next_full",
            use_phase_velocity=True,
            phase_velocity_hidden=8,
            phase_velocity_scale=0.1,
            phase_use_circular_attn_bias=True,
            phase_circular_attn_bias_scale=1.0,
            use_weak_period_residual=True,
            weak_period_residual_gate_init=0.5,
            use_phase_local_trend=True,
            phase_local_trend_window=3,
            phase_local_trend_gate_init=0.0,
            use_adaptive_residual_gate=True,
            adaptive_residual_gate_hidden=8,
            adaptive_residual_gate_init=0.5,
        )
    # Pure-phase plan mechanisms (Adaptive Phase Geometry Forecasting).
    # Each module is a warm-start identity and independently ablable; pure_full
    # stacks all four with residual reconstruction disabled so every gain comes
    # from the phase path.
    if mode == "multiscale_phase":
        return dict(
            scheme_name="multiscale_phase",
            use_multiscale_phase=True,
            phase_multiscale_long_period=48,
            phase_multiscale_coarse=2,
        )
    if mode == "phase_deformation":
        return dict(
            scheme_name="phase_deformation",
            use_phase_deformation=True,
            phase_deformation_hidden=8,
            phase_deformation_scale=0.2,
        )
    if mode == "phase_geo":
        return dict(
            scheme_name="phase_geo",
            phase_use_circular_attn_bias=True,
            phase_circular_attn_bias_scale=1.0,
        )
    if mode == "phase_graph":
        return dict(
            scheme_name="phase_graph",
            use_phase_graph=True,
            phase_graph_hidden=16,
            phase_graph_k=2,
        )
    if mode == "predictor_mlp":
        return dict(
            scheme_name="predictor_mlp",
            predictor_use_mlp=True,
            predictor_dropout=0.0,
        )
    if mode == "trajectory_decoder":
        return dict(
            scheme_name="trajectory_decoder",
            use_trajectory_decoder=True,
            phase_decoder_hidden=64,
            phase_decoder_order=2,
        )
    if mode == "pure_full":
        return dict(
            scheme_name="pure_full",
            use_residual_head=False,
            use_multiscale_phase=True,
            phase_multiscale_long_period=48,
            phase_multiscale_coarse=2,
            use_phase_deformation=True,
            phase_deformation_hidden=8,
            phase_deformation_scale=0.2,
            use_phase_graph=True,
            phase_graph_hidden=16,
            phase_graph_k=2,
            use_trajectory_decoder=True,
            phase_decoder_hidden=64,
            phase_decoder_order=2,
        )
    # Residual-topology modes.  No mode below enables phase-local trend or a
    # dynamic-phase module, so the only experimental variable is the residual
    # path and its insertion point.
    if mode == "residual_output_convex":
        return dict(
            scheme_name="residual_output_convex",
            use_topology_output_convex_residual=True,
            topology_output_convex_gate_init=0.5,
        )
    if mode == "residual_output_additive":
        return dict(
            scheme_name="residual_output_additive",
            use_additive_output_residual=True,
            additive_output_residual_gate_init=0.5,
        )
    if mode == "residual_latent_long":
        return dict(
            scheme_name="residual_latent_long",
            use_latent_long_residual=True,
        )
    if mode == "residual_latent_layerwise":
        return dict(
            scheme_name="residual_latent_layerwise",
            use_layerwise_latent_residual=True,
        )
    if mode == "residual_hybrid":
        return dict(
            scheme_name="residual_hybrid",
            use_layerwise_latent_residual=True,
            use_additive_output_residual=True,
            additive_output_residual_gate_init=0.5,
        )
    # Layer-wise output residuals.  Each enables the single-point parent
    # (R1 convex / R2 additive) at the final output plus the same fusion form at
    # every intermediate routing layer.  On 1-layer models they reduce to the
    # parent; the convex intermediate gate starts closed (gate_init 0.0 ->
    # clamped 1e-4) so construction-time output is (near-)identical to the
    # parent.
    if mode == "residual_output_layerwise_convex":
        return dict(
            scheme_name="residual_output_layerwise_convex",
            use_topology_output_convex_residual=True,
            topology_output_convex_gate_init=0.5,
            use_layerwise_output_convex=True,
            layerwise_output_convex_gate_init=0.0,
        )
    if mode == "residual_output_layerwise_additive":
        return dict(
            scheme_name="residual_output_layerwise_additive",
            use_additive_output_residual=True,
            additive_output_residual_gate_init=0.5,
            use_layerwise_output_additive=True,
            layerwise_output_additive_gate_init=0.5,
        )
    # Golden-combo modes (gold_combo_stability_v1).  All four share the phase
    # stack frozen in the plan: uncertainty min 0.2 / trend gate 0.05,
    # period-level gate 0.2 / slope gate 0.05, high-frequency 0.8 / 0.5 / w7.
    # The shared residual gate prior is alpha_0 = 0.5; only the output fusion
    # differs (fixed / existing 3-feature MLP gate / RCRF at two sensitivities).
    if mode in ("gold_combo_fixed", "gold_combo_adaptive",
                "gold_combo_reliability_s0", "gold_combo_reliability_s2"):
        overrides = {
            "use_phase_uncertainty_shrinkage": True,
            "phase_uncertainty_min": 0.2,
            "phase_uncertainty_trend_gate_init": 0.05,
            "use_phase_period_level_calibration": True,
            "phase_level_slope_window": 3,
            "phase_level_slope_gate_init": 0.05,
            "phase_level_calib_gate_init": 0.2,
            "use_phase_noise_hifreq_damping": True,
            "phase_noise_hifreq_strength": 0.8,
            "phase_noise_hifreq_threshold": 0.5,
            "phase_noise_hifreq_temperature": 0.2,
            "phase_noise_hifreq_window": 7,
            "use_weak_period_residual": True,
            "weak_period_residual_gate_init": 0.5,
        }
        if mode == "gold_combo_fixed":
            overrides["scheme_name"] = mode
            return overrides
        if mode == "gold_combo_adaptive":
            overrides["scheme_name"] = mode
            overrides["use_adaptive_weak_period_gate"] = True
            return overrides
        sensitivity = 0.0 if mode == "gold_combo_reliability_s0" else 2.0
        overrides["scheme_name"] = mode
        overrides["use_rcrf_fusion"] = True
        overrides["rcrf_alpha_init"] = 0.5
        overrides["rcrf_sensitivity_init"] = sensitivity
        overrides["rcrf_s_max"] = 4.0
        return overrides
    if mode in PERIODIC_RESIDUAL_PE_MODES:
        # Controlled extension of the frozen RCRF candidate: the phase stack,
        # RCRF formula, NLinear map and all training settings stay unchanged.
        overrides = get_ablation_overrides("gold_combo_reliability_s2")
        overrides.update(
            scheme_name=mode,
            weak_period_residual_head_type="periodic_pe",
            use_periodic_residual_pe=True,
            periodic_residual_pe_type=PERIODIC_RESIDUAL_PE_MODES[mode],
            periodic_residual_pe_dim=16,
            periodic_residual_pe_temperature=0.1,
            periodic_residual_pe_cycle_decay=0.1,
            periodic_residual_pe_blend_init=0.1,
        )
        return overrides
    # Inter-Cycle Patch Transformer candidates (ICPT plan).  A3/A4 swap the
    # NLinear head for simple cycle baselines; rcrf_icpt_* build the ICPT head
    # with a position encoding.  Everything else inherits the frozen RCRF stack
    # unchanged so the only structural difference from A2 is NLinear -> ICPT.
    if mode == "rcrf_repeat_last_cycle":
        overrides = get_ablation_overrides("gold_combo_reliability_s2")
        overrides.update(
            scheme_name=mode, weak_period_residual_head_type="repeat_last_cycle"
        )
        return overrides
    if mode == "rcrf_cycle_net":
        overrides = get_ablation_overrides("gold_combo_reliability_s2")
        overrides.update(scheme_name=mode, weak_period_residual_head_type="cycle_net")
        return overrides
    if mode in INTERCYCLE_PE_MODES:
        overrides = get_ablation_overrides("gold_combo_reliability_s2")
        overrides.update(
            scheme_name=mode, **ICPT_FIXED_HYPERPARAMS,
            intercycle_pe_type=INTERCYCLE_PE_MODES[mode],
        )
        return overrides
    if mode in INTERCYCLE_HORIZON_PE_MODES:
        overrides = get_ablation_overrides("gold_combo_reliability_s2")
        overrides.update(
            scheme_name=mode,
            **ICPT_HORIZON_FIXED_HYPERPARAMS,
            intercycle_pe_type=INTERCYCLE_HORIZON_PE_MODES[mode],
        )
        return overrides
    if mode == "rcrf_icpt_horizon_cycle_anchor":
        overrides = get_ablation_overrides("rcrf_icpt_horizon_none")
        overrides.update(
            scheme_name=mode,
            intercycle_anchor_mode="last_cycle",
            intercycle_use_last_cycle_anchor=True,
        )
        return overrides
    # Stage D mechanism ablations of the frozen ICPT-best candidate.
    # The frozen index-PE is resolved from the ICPT_FROZEN_PE environment
    # variable (set by the experiment runner after Stage B freezes), so the
    # same preset name builds with the frozen PE inside subprocess runners.
    # The default keeps none so presets are usable in unit tests before any
    # freeze exists.
    if mode in ("icpt_only", "icpt_fixed_fusion", "icpt_patch16",
                "icpt_no_anchor", "icpt_no_attention"):
        overrides = get_ablation_overrides("rcrf_icpt_none")
        pe = os.environ.get("ICPT_FROZEN_PE") or "none"
        overrides["intercycle_pe_type"] = pe
        overrides["scheme_name"] = mode
        if mode == "icpt_only":
            # Single-branch: ICPT alone, no phase fusion (gate pinned ~1).
            overrides.update(use_rcrf_fusion=False, weak_period_residual_gate_init=1.0)
        elif mode == "icpt_fixed_fusion":
            # PhaseFormer + fixed 0.5 blend of the ICPT residual.
            overrides.update(use_rcrf_fusion=False, weak_period_residual_gate_init=0.5)
        elif mode == "icpt_patch16":
            # Non-period-aligned 16-length patch tokens instead of 24-length cycles.
            overrides["intercycle_period_len"] = 16
        elif mode == "icpt_no_anchor":
            overrides["intercycle_use_last_cycle_anchor"] = False
        elif mode == "icpt_no_attention":
            overrides["intercycle_use_attention"] = False
        return overrides
    raise ValueError(f"Unsupported ablation mode: {mode}")


def _without_residual(overrides):
    sanitized = dict(overrides)
    sanitized["use_weak_period_residual"] = False
    sanitized["use_adaptive_weak_period_gate"] = False
    sanitized["use_periodic_residual_pe"] = False
    return sanitized


def get_best_nonresidual_overrides(dataset_name, horizon):
    """Per-dataset/horizon best phase-only policy from the full ablation suite.

    The policy intentionally excludes weak/adaptive residual mechanisms. Choices
    are selected by test MSE from phaseformer_ablation_full_20260716.
    """
    selected = {
        ("ETTh1", 96): "phase_all",
        ("ETTh1", 192): "phase_level",
        ("ETTh1", 336): "phase_uncertainty",
        ("ETTh1", 720): "original",
        ("ETTh2", 96): "phase_uncertainty",
        ("ETTh2", 192): "phase_uncertainty",
        ("ETTh2", 336): "phase_uncertainty",
        ("ETTh2", 720): "phase_all",
        ("ETTm1", 96): "phase_uncertainty",
        ("ETTm1", 192): "phase_hifreq",
        ("ETTm1", 336): "phase_hifreq",
        ("ETTm1", 720): "phase_all",
        ("ETTm2", 96): "phase_all",
        ("ETTm2", 192): "phase_uncertainty",
        ("ETTm2", 336): "phase_level",
        ("ETTm2", 720): "phase_uncertainty",
        ("Weather", 96): "phase_all",
        ("Weather", 192): "phase_level",
        ("Weather", 336): "phase_level",
        ("Weather", 720): "phase_all",
        ("Electricity", 96): "time_mark",
        ("Electricity", 192): "time_mark",
        ("Electricity", 336): "time_mark",
        ("Electricity", 720): "time_mark",
    }
    mode = selected.get((dataset_name, horizon), "original")
    if mode == "original":
        return _without_residual(
            dict(scheme_name=f"best_nonresidual_{dataset_name.lower()}_{horizon}_original")
        )
    overrides = get_ablation_overrides(mode)
    overrides["scheme_name"] = f"best_nonresidual_{dataset_name.lower()}_{horizon}_{mode}"
    return _without_residual(overrides)


def make_exp_args(dataset_name, lookback, horizon, hyperparams, batch_size=None):
    exp_args = deepcopy(config_module.config)
    exp_args.model_args.model = "PhaseFormer"
    exp_args.model_args.input_len = exp_args.dataset_args.seq_len = lookback
    exp_args.model_args.num_variants = int(DATASET_INFO[dataset_name]["num_variants"])

    exp_args.training_args.itr = 1
    exp_args.training_args.patience = hyperparams.get("patience", 8)
    exp_args.training_args.ema = False
    exp_args.training_args.train_epochs = hyperparams.get("train_epochs", 30)
    exp_args.training_args.lr_schedule_config.type = "type3"
    loss_func = str(hyperparams.get("loss_func", "mse")).lower()
    legacy_use_huber = hyperparams.get("use_huber_loss", True)
    if legacy_use_huber:
        loss_func = "huber"
    exp_args.training_args.loss_func = loss_func
    # Retained as a compatibility mirror for old configs and result schemas.
    exp_args.training_args.use_huber_loss = loss_func == "huber"
    exp_args.training_args.huber_delta = hyperparams.get("huber_delta", 1.0)
    exp_args.training_args.learning_rate = hyperparams["learning_rate"]
    exp_args.training_args.batch_size = (
        batch_size or hyperparams.get("batch_size") or DATASET_INFO[dataset_name]["batch_size"]
    )

    exp_args.dataset_args.percent = 100
    exp_args.dataset_args.data = DATASET_INFO[dataset_name]["data"]
    exp_args.dataset_args.root_path = DATASET_INFO[dataset_name]["root_path"]
    exp_args.dataset_args.data_path = DATASET_INFO[dataset_name]["data_path"]
    exp_args.dataset_args.freq = get_frequency(dataset_name)
    exp_args.dataset_args.batch_size = exp_args.training_args.batch_size
    exp_args.dataset_args.seq_len = lookback
    exp_args.dataset_args.pred_len = horizon
    exp_args.dataset_args.noisy_ratio = 0.0
    exp_args.dataset_args.var_needed = exp_args.model_args.num_variants
    return exp_args


class PhaseFormerPresetConfig:
    def __init__(self, exp_args, lookback, horizon, hyperparams):
        self.seq_len = lookback
        self.pred_len = horizon
        self.enc_in = exp_args.model_args.num_variants
        self.period_len = hyperparams.get("period_len", 24)
        self.target_var_index = -1
        self.training_args = exp_args.training_args
        self.dataset_args = exp_args.dataset_args

        self.latent_dim = hyperparams["latent_dim"]
        self.phase_encoder_hidden = hyperparams["phase_encoder_hidden"]
        self.predictor_hidden = hyperparams["predictor_hidden"]
        self.phase_layers = hyperparams["layers"]
        self.phase_attn_heads = hyperparams["phase_attn_heads"]
        self.phase_attn_dropout = hyperparams.get("phase_attn_dropout", 0.1)
        self.phase_attn_use_relpos = hyperparams.get("phase_attn_use_relpos", True)
        self.phase_attn_window = hyperparams.get("phase_attn_window", None)
        self.phase_attention_dim = hyperparams.get("phase_attention_dim", None)
        self.phase_num_routers = hyperparams["phase_num_routers"]
        self.phase_use_pos_embed = hyperparams.get("phase_use_pos_embed", True)
        self.phase_pos_dropout = hyperparams.get("phase_pos_dropout", 0.0)

        self.use_revin = hyperparams.get("use_revin", True)
        self.revin_affine = hyperparams.get("revin_affine", DEFAULT_NORM_HYPERS["revin_affine"])
        self.revin_eps = hyperparams.get("revin_eps", DEFAULT_NORM_HYPERS["revin_eps"])
        self.use_huber_loss = exp_args.training_args.use_huber_loss
        self.huber_delta = exp_args.training_args.huber_delta

        self.use_weak_period_residual = hyperparams.get("use_weak_period_residual", False)
        self.weak_period_residual_gate_init = hyperparams.get(
            "weak_period_residual_gate_init", 0.2
        )
        self.weak_period_residual_head_type = hyperparams.get(
            "weak_period_residual_head_type", "shared"
        )
        self.use_adaptive_weak_period_gate = hyperparams.get(
            "use_adaptive_weak_period_gate", False
        )
        self.adaptive_weak_period_gate_hidden = hyperparams.get(
            "adaptive_weak_period_gate_hidden", 8
        )
        # RCRF (gold_combo_stability_v1): reliability-coupled convex fusion gate.
        self.use_rcrf_fusion = hyperparams.get("use_rcrf_fusion", False)
        self.rcrf_alpha_init = hyperparams.get("rcrf_alpha_init", 0.5)
        self.rcrf_sensitivity_init = hyperparams.get("rcrf_sensitivity_init", 0.0)
        self.rcrf_s_max = hyperparams.get("rcrf_s_max", 4.0)
        self.rcrf_eps = hyperparams.get("rcrf_eps", 1e-6)
        self.use_periodic_residual_pe = hyperparams.get(
            "use_periodic_residual_pe", False
        )
        self.periodic_residual_pe_type = hyperparams.get(
            "periodic_residual_pe_type", "harmonic"
        )
        self.periodic_residual_pe_dim = hyperparams.get(
            "periodic_residual_pe_dim", 16
        )
        self.periodic_residual_pe_temperature = hyperparams.get(
            "periodic_residual_pe_temperature", 0.1
        )
        self.periodic_residual_pe_cycle_decay = hyperparams.get(
            "periodic_residual_pe_cycle_decay", 0.1
        )
        self.periodic_residual_pe_blend_init = hyperparams.get(
            "periodic_residual_pe_blend_init", 0.1
        )
        # Inter-Cycle Patch Transformer residual head configuration (ICPT plan).
        self.intercycle_period_len = hyperparams.get("intercycle_period_len", 24)
        self.intercycle_d_model = hyperparams.get("intercycle_d_model", 32)
        self.intercycle_heads = hyperparams.get("intercycle_heads", 4)
        self.intercycle_ffn_dim = hyperparams.get("intercycle_ffn_dim", 64)
        self.intercycle_encoder_layers = hyperparams.get(
            "intercycle_encoder_layers", 1
        )
        self.intercycle_decoder_layers = hyperparams.get(
            "intercycle_decoder_layers", 1
        )
        self.intercycle_pe_type = hyperparams.get("intercycle_pe_type", "none")
        self.intercycle_relative_buckets = hyperparams.get(
            "intercycle_relative_buckets", 16
        )
        self.intercycle_lff_frequencies = hyperparams.get(
            "intercycle_lff_frequencies", 16
        )
        self.intercycle_use_last_cycle_anchor = hyperparams.get(
            "intercycle_use_last_cycle_anchor", True
        )
        self.intercycle_use_attention = hyperparams.get(
            "intercycle_use_attention", True
        )
        self.intercycle_dropout = hyperparams.get("intercycle_dropout", 0.0)
        self.intercycle_prediction_head = hyperparams.get(
            "intercycle_prediction_head", "decoder"
        )
        self.intercycle_anchor_mode = hyperparams.get(
            "intercycle_anchor_mode", None
        )
        self.use_time_mark_adjustment = hyperparams.get("use_time_mark_adjustment", False)
        self.time_mark_dim = hyperparams.get("time_mark_dim", 4)
        self.time_mark_hidden = hyperparams.get("time_mark_hidden", 32)
        self.use_phase_local_trend = hyperparams.get("use_phase_local_trend", False)
        self.phase_local_trend_window = hyperparams.get("phase_local_trend_window", 3)
        self.phase_local_trend_gate_init = hyperparams.get(
            "phase_local_trend_gate_init", 0.0
        )
        self.use_phase_uncertainty_shrinkage = hyperparams.get(
            "use_phase_uncertainty_shrinkage", False
        )
        self.phase_uncertainty_min = hyperparams.get("phase_uncertainty_min", 0.35)
        self.phase_uncertainty_trend_gate_init = hyperparams.get(
            "phase_uncertainty_trend_gate_init", 0.05
        )
        self.use_phase_period_level_calibration = hyperparams.get(
            "use_phase_period_level_calibration", False
        )
        self.phase_level_slope_window = hyperparams.get("phase_level_slope_window", 3)
        self.phase_level_slope_gate_init = hyperparams.get(
            "phase_level_slope_gate_init", 0.05
        )
        self.phase_level_calib_gate_init = hyperparams.get(
            "phase_level_calib_gate_init", 0.1
        )
        self.use_phase_noise_hifreq_damping = hyperparams.get(
            "use_phase_noise_hifreq_damping", False
        )
        self.phase_noise_hifreq_strength = hyperparams.get(
            "phase_noise_hifreq_strength", 0.5
        )
        self.phase_noise_hifreq_threshold = hyperparams.get(
            "phase_noise_hifreq_threshold", 1.0
        )
        self.phase_noise_hifreq_temperature = hyperparams.get(
            "phase_noise_hifreq_temperature", 0.2
        )
        self.phase_noise_hifreq_window = hyperparams.get("phase_noise_hifreq_window", 7)
        self.use_phase_align = hyperparams.get("use_phase_align", False)
        self.phase_align_hidden = hyperparams.get("phase_align_hidden", 8)
        self.phase_align_mark_dim = hyperparams.get("phase_align_mark_dim", None)
        self.phase_align_position_encoding = hyperparams.get(
            "phase_align_position_encoding", False
        )
        self.phase_align_chunk = hyperparams.get("phase_align_chunk", 240)
        self.use_phase_warp = hyperparams.get("use_phase_warp", False)
        self.phase_warp_hidden = hyperparams.get("phase_warp_hidden", 8)
        self.phase_warp_mark_dim = hyperparams.get("phase_warp_mark_dim", None)
        self.phase_warp_chunk = hyperparams.get("phase_warp_chunk", 240)
        self.use_phase_amp_calib = hyperparams.get("use_phase_amp_calib", False)
        self.phase_amp_calib_hidden = hyperparams.get("phase_amp_calib_hidden", 8)
        self.phase_amp_calib_max_scale = hyperparams.get(
            "phase_amp_calib_max_scale", 2.0
        )
        self.use_phase_rape = hyperparams.get("use_phase_rape", False)
        self.phase_rape_gate_hidden = hyperparams.get("phase_rape_gate_hidden", 8)
        self.phase_rape_mark_dim = hyperparams.get("phase_rape_mark_dim", None)

        # Dynamic-phase mechanism flags (experiment plan stages 1-5).
        self.use_residual_head = hyperparams.get("use_residual_head", True)
        self.phase_use_circular_pos = hyperparams.get("phase_use_circular_pos", False)
        self.phase_use_circular_attn_bias = hyperparams.get(
            "phase_use_circular_attn_bias", False
        )
        self.phase_circular_attn_bias_scale = hyperparams.get(
            "phase_circular_attn_bias_scale", 1.0
        )
        self.use_phase_correction = hyperparams.get("use_phase_correction", False)
        self.phase_correction_hidden = hyperparams.get(
            "phase_correction_hidden", self.latent_dim
        )
        self.use_phase_rotation = hyperparams.get("use_phase_rotation", False)
        self.phase_rotation_hidden = hyperparams.get("phase_rotation_hidden", 8)
        self.use_harmonic_modulation = hyperparams.get("use_harmonic_modulation", False)
        self.harmonic_modulation_hidden = hyperparams.get(
            "harmonic_modulation_hidden", 8
        )
        self.harmonic_modulation_max_scale = hyperparams.get(
            "harmonic_modulation_max_scale", 2.0
        )
        # Next-stage paper plan mechanism flags (stages 1 and 3).
        self.use_phase_velocity = hyperparams.get("use_phase_velocity", False)
        self.phase_velocity_hidden = hyperparams.get("phase_velocity_hidden", 8)
        self.phase_velocity_scale = hyperparams.get("phase_velocity_scale", 0.1)
        self.use_adaptive_residual_gate = hyperparams.get(
            "use_adaptive_residual_gate", False
        )
        self.adaptive_residual_gate_hidden = hyperparams.get(
            "adaptive_residual_gate_hidden", 8
        )
        self.adaptive_residual_gate_init = hyperparams.get(
            "adaptive_residual_gate_init", 0.5
        )
        self.use_phase_sparse_event_calibration = hyperparams.get(
            "use_phase_sparse_event_calibration", False
        )
        self.phase_sparse_event_window = hyperparams.get("phase_sparse_event_window", 3)
        self.phase_sparse_event_gate_init = hyperparams.get(
            "phase_sparse_event_gate_init", 0.05
        )
        self.phase_sparse_event_max_boost = hyperparams.get(
            "phase_sparse_event_max_boost", 1.0
        )
        self.phase_sparse_event_temperature = hyperparams.get(
            "phase_sparse_event_temperature", 0.2
        )
        # Pure-phase plan mechanism flags (stages 1-4) + predictor head type.
        self.predictor_use_mlp = hyperparams.get("predictor_use_mlp", False)
        self.predictor_dropout = hyperparams.get("predictor_dropout", 0.0)
        self.use_multiscale_phase = hyperparams.get("use_multiscale_phase", False)
        self.phase_multiscale_long_period = hyperparams.get(
            "phase_multiscale_long_period", 2 * self.period_len
        )
        self.phase_multiscale_coarse = hyperparams.get("phase_multiscale_coarse", 2)
        self.use_phase_deformation = hyperparams.get("use_phase_deformation", False)
        self.phase_deformation_hidden = hyperparams.get(
            "phase_deformation_hidden", 8
        )
        self.phase_deformation_scale = hyperparams.get(
            "phase_deformation_scale", 0.2
        )
        self.use_phase_graph = hyperparams.get("use_phase_graph", False)
        self.phase_graph_hidden = hyperparams.get("phase_graph_hidden", 16)
        self.phase_graph_k = hyperparams.get("phase_graph_k", 2)
        self.use_trajectory_decoder = hyperparams.get(
            "use_trajectory_decoder", False
        )
        # Residual-topology experiment flags.
        self.use_additive_output_residual = hyperparams.get(
            "use_additive_output_residual", False
        )
        self.use_topology_output_convex_residual = hyperparams.get(
            "use_topology_output_convex_residual", False
        )
        self.topology_output_convex_gate_init = hyperparams.get(
            "topology_output_convex_gate_init", 0.5
        )
        self.additive_output_residual_gate_init = hyperparams.get(
            "additive_output_residual_gate_init", 0.5
        )
        self.use_latent_long_residual = hyperparams.get(
            "use_latent_long_residual", False
        )
        self.use_layerwise_latent_residual = hyperparams.get(
            "use_layerwise_latent_residual", False
        )
        self.use_layerwise_output_convex = hyperparams.get(
            "use_layerwise_output_convex", False
        )
        self.use_layerwise_output_additive = hyperparams.get(
            "use_layerwise_output_additive", False
        )
        self.layerwise_output_convex_gate_init = hyperparams.get(
            "layerwise_output_convex_gate_init", 0.0
        )
        self.layerwise_output_additive_gate_init = hyperparams.get(
            "layerwise_output_additive_gate_init", 0.5
        )
        self.phase_decoder_hidden = hyperparams.get("phase_decoder_hidden", 64)
        self.phase_decoder_order = hyperparams.get("phase_decoder_order", 2)

    def get(self, key, default=None):
        return getattr(self, key, default)


def build_hyperparams(dataset_name, horizon, mode):
    hyperparams = get_base_hyperparams(dataset_name, horizon)
    hyperparams["scheme_name"] = "original"
    if mode == "latest":
        hyperparams.update(get_latest_overrides(dataset_name, horizon))
    elif mode == "best_nonresidual":
        hyperparams.update(get_best_nonresidual_overrides(dataset_name, horizon))
    elif mode in ABLATION_MODES:
        hyperparams.update(get_ablation_overrides(mode))
    elif mode != "original":
        raise ValueError(f"Unsupported mode: {mode}")
    return hyperparams
