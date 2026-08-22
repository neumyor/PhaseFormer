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
    raise ValueError(f"Unsupported ablation mode: {mode}")


def _without_residual(overrides):
    sanitized = dict(overrides)
    sanitized["use_weak_period_residual"] = False
    sanitized["use_adaptive_weak_period_gate"] = False
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
