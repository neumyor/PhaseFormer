import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pl_bases.default_module import DefaultPLModule
from src.models.iTransformer import iTransformer
from src.models.PhaseBase_0 import (
    RevIN,
    PhaseSeriesEncoder,
    DimensionReductionAttention,
    PhasePredictor,
)


class PhaseFormer(DefaultPLModule):
    """
    PhaseFormer: iTransformer for cross-channel fusion -> Phase-based enhancement.

    Pipeline:
    1) RevIN over time per variable
    2) iTransformer reconstructs/refines the historical sequence (length = seq_len)
       to fuse cross-channel information in a data-driven way
    3) Phase modules (encoder -> cross-phase attention -> predictor) generate future phases
    4) Reassemble to forecasting sequence and de-normalize
    """

    def __init__(self, configs):
        super().__init__(configs)

        # basics
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        # phase dims
        self.latent_dim = getattr(configs, "latent_dim", 8)
        self.phase_encoder_hidden = getattr(configs, "phase_encoder_hidden", 32)
        self.predictor_hidden = getattr(configs, "predictor_hidden", 64)

        # phase attention params
        self.phase_attn_heads = getattr(configs, "phase_attn_heads", 4)
        self.phase_attn_dropout = getattr(configs, "phase_attn_dropout", 0.0)
        self.phase_attn_use_relpos = getattr(configs, "phase_attn_use_relpos", True)
        self.phase_attn_window = getattr(configs, "phase_attn_window", None)
        self.phase_attention_dim = getattr(configs, "phase_attention_dim", None)
        self.phase_num_routers = getattr(configs, "phase_num_routers", 8)
        self.phase_use_pos_embed = getattr(configs, "phase_use_pos_embed", False)
        self.phase_pos_dropout = getattr(configs, "phase_pos_dropout", 0.0)

        # period calculations
        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        # normalization
        self.use_revin = getattr(configs, "use_revin", True)
        self.revin_affine = getattr(configs, "revin_affine", False)
        self.revin_eps = getattr(configs, "revin_eps", 1e-5)
        if self.use_revin:
            self.revin = RevIN(num_features=self.enc_in, eps=self.revin_eps, affine=self.revin_affine)

        # loss config: support Huber loss via switch
        self.use_huber_loss = getattr(configs, "use_huber_loss", False)
        self.huber_delta = getattr(configs, "huber_delta", 1.0)

        # --- iTransformer for cross-channel fusion ---
        itrans_configs = copy.deepcopy(configs)
        # set projector output length to seq_len for history reconstruction
        itrans_configs.pred_len = self.seq_len
        # avoid iTransformer's internal non-stationary normalization; we use RevIN outside
        itrans_configs.use_norm = False
        # optional: allow skipping attention via config flag `itransformer_skip_attn`
        skip_attn_flag = getattr(configs, "itransformer_skip_attn", False)
        try:
            itrans_configs.skip_attn = skip_attn_flag
        except Exception:
            pass
        self.itransformer = iTransformer(itrans_configs)

        # --- phase modules (reusing PhaseBase components) ---
        self.phase_encoder = PhaseSeriesEncoder(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim,
            hidden=self.phase_encoder_hidden,
            use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
            dropout=getattr(configs, "phase_encoder_dropout", 0.0),
        )

        self.phase_interact = DimensionReductionAttention(
            latent_dim=self.latent_dim,
            num_routers=self.phase_num_routers,
            num_heads=self.phase_attn_heads,
            dropout=self.phase_attn_dropout,
            use_relpos=self.phase_attn_use_relpos,
            period_len=self.period_len,
            window_size=self.phase_attn_window,
            attention_dim=self.phase_attention_dim,
            use_pos_embed=self.phase_use_pos_embed,
            pos_dropout=self.phase_pos_dropout,
        )

        self.predictor = PhasePredictor(
            latent_dim=self.latent_dim,
            out_steps=self.num_periods_output,
            hidden=self.predictor_hidden,
            use_mlp=getattr(configs, "predictor_use_mlp", False),
            dropout=getattr(configs, "predictor_dropout", 0.0),
        )

    # ---------- helpers for phase rearrangement ----------
    @staticmethod
    def _to_phase_series(x_periods):
        """(B, C, P_in, L) -> (B, C, L, P_in)"""
        return x_periods.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _from_phase_steps_to_periods(y_phase_steps):
        """(B, C, L, P_out) -> (B, C, P_out, L)"""
        return y_phase_steps.permute(0, 1, 3, 2).contiguous()

    def _fuse_cross_channel_with_itransformer(self, x_enc, x_mark_enc=None):
        """
        Use iTransformer to reconstruct/refine the historical window with cross-channel fusion.
        Input: x_enc (B, L, C) -> Output: x_refined (B, L, C)
        """
        # iTransformer returns shape (B, seq_len, C) with our pred_len override
        with torch.no_grad() if getattr(self.args.training_args, "freeze_itransformer", False) else torch.enable_grad():
            refined = self.itransformer(x_enc, x_mark_enc)  # (B, seq_len, C)
        return refined

    # ------------- forward --------------
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        Input:  x_enc (B, seq_len, C)
        Output: y_hat (B, pred_len, C)
        Also returns intermediate Z (B,C,L,D) and future phase values (B,C,L,P_out) for analysis.
        """
        # 1) RevIN
        if self.use_revin:
            x_in, stats = self.revin.normalize(x_enc.float())  # (B, L, C)
        else:
            x_in, stats = x_enc.float(), None

        # ensure time features are float32 to avoid dtype mismatch inside linear layers
        x_mark_fp32 = x_mark_enc.float() if x_mark_enc is not None else None

        # 2) iTransformer cross-channel fusion (reconstruct window)
        x_fused = self._fuse_cross_channel_with_itransformer(x_in, x_mark_fp32)  # (B, L, C)

        # 3) Ring padding to full periods
        x = x_fused.permute(0, 2, 1)  # (B, C, L_total)
        B, C, L = x.shape
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode="circular")  # (B, C, total_len_in)

        # 4) Split to (B, C, P_in, L)
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)

        # 5) Parallel by phase: take columns (B, C, L, P_in)
        phase_series = self._to_phase_series(x_periods)

        # 6) Encode -> (B, C, L, D)
        Z = self.phase_encoder(phase_series)

        # 7) Cross-phase interaction
        Z = self.phase_interact(Z)

        # 8) Predict next P_out for each phase step
        y_phase_steps = self.predictor(Z)  # (B, C, L, P_out)

        # 9) Reassemble to forecasting sequence
        y_periods = self._from_phase_steps_to_periods(y_phase_steps)  # (B, C, P_out, L)
        y_full = y_periods.reshape(B, C, -1)[..., : self.pred_len]  # (B, C, pred_len)
        y_hat = y_full.permute(0, 2, 1)  # (B, pred_len, C)

        # 10) De-normalization
        if stats is not None:
            y_hat = self.revin.denormalize(y_hat, stats)

        return y_hat, Z, y_phase_steps

    # --------- Lightning steps (align with PhaseBase) ----------
    def _compute_loss(self, outputs, target):
        """Select loss: Huber if enabled, otherwise fallback to configured criterion."""
        use_huber = self.use_huber_loss or str(getattr(self.args.training_args, "loss_func", "")).lower() == "huber"
        if use_huber:
            # custom Huber to control delta across torch versions
            diff = outputs - target
            abs_diff = torch.abs(diff)
            delta = torch.as_tensor(self.huber_delta, device=outputs.device, dtype=outputs.dtype)
            quadratic = torch.minimum(abs_diff, delta)
            linear = abs_diff - quadratic
            loss = 0.5 * (quadratic ** 2) / delta + linear
            return loss.mean()
        else:
            criterion = self._get_criterion(self.args.training_args.loss_func)
            return criterion(outputs, target)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        # for compatibility with DefaultPLModule interfaces
        dec_inp = self._build_decoder_input(batch_y)

        outputs, Z, _ = self(
            x_enc=batch_x, x_mark_enc=batch_x_mark, x_dec=dec_inp, x_mark_dec=batch_y_mark
        )

        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        loss = self._compute_loss(outputs, target)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

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


