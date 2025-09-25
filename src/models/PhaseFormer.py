import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pl_bases.default_module import DefaultPLModule
from src.models.PhaseFormer_old import (
    RevIN,
    PhaseSeriesEncoder,
    DimensionReductionAttention,
)

class PhaseSeriesDecoder(nn.Module):
    """Simple decoder that maps from latent dimension to output periods using a single linear layer."""
    
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


class PhaseFormerBlock(nn.Module):
    """One stackable layer: Encoder -> Interaction -> Decoder.

    The decoder's out_steps is configurable so that intermediate blocks can
    preserve the same number of phase steps as the input (P_in), while the
    final block can output a different number of steps (P_out).
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

        self.encoder = PhaseSeriesEncoder(
            p_in=num_periods_input,
            latent_dim=latent_dim,
            hidden=phase_encoder_hidden,
            use_mlp=phase_encoder_use_mlp,
            dropout=phase_encoder_dropout,
        )

        self.interact = DimensionReductionAttention(
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

        self.decoder = PhaseSeriesDecoder(
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
    PhaseFormer: Phase-based enhancement without external cross-channel fusion.

    Pipeline:
    1) RevIN over time per variable
    2) Phase modules (encoder -> cross-phase attention -> predictor) generate future phases
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

        # loss configuration
        self.use_huber_loss = getattr(configs, "use_huber_loss", False)
        self.huber_delta = getattr(configs, "huber_delta", 1.0)

        # stackable phase blocks (Encoder -> Interaction -> Decoder)
        self.phase_layers = getattr(configs, "phase_layers", 1)
        blocks = []
        for li in range(self.phase_layers):
            out_steps = self.num_periods_input if li < self.phase_layers - 1 else self.num_periods_output
            blocks.append(
                PhaseFormerBlock(
                    num_periods_input=self.num_periods_input,
                    num_periods_output=out_steps,
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
                    phase_encoder_use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
                    phase_encoder_dropout=getattr(configs, "phase_encoder_dropout", 0.0),
                    predictor_use_mlp=getattr(configs, "predictor_use_mlp", False),
                    predictor_dropout=getattr(configs, "predictor_dropout", 0.0),
                )
            )
        self.blocks = nn.ModuleList(blocks)

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

        # 5) Parallel by phase (B, C, L, P_in)
        phase_series = self._to_phase_series(x_periods)

        # 6-8) Stacked PhaseFormer blocks
        Z = None
        y_phase_steps = None
        phase_series_cur = phase_series
        for layer_index, block in enumerate(self.blocks):
            Z, y_phase_steps = block(phase_series_cur, Z)
            # For intermediate layers, keep the same steps length as input (P_in)
            # and feed it to the next layer. The last layer outputs P_out and is
            # not fed forward.
            if layer_index < len(self.blocks) - 1:
                phase_series_cur = y_phase_steps  # shape: (B, C, L, P_in)

        # 9) Reassemble to sequence (B, pred_len, C)
        y_periods = self._from_phase_steps_to_periods(y_phase_steps)  # (B, C, P_out, L)
        y_full = y_periods.reshape(B, C, -1)[..., : self.pred_len]  # (B, C, pred_len)
        y_hat = y_full.permute(0, 2, 1)  # (B, pred_len, C)

        # 10) De-normalization
        if self.use_revin:
            y_hat = self.revin.denormalize(y_hat, stats)

        return y_hat, Z, y_phase_steps

    # Lightning training steps
    def _compute_loss(self, outputs, target):
        """Loss computation with Huber support."""
        use_huber = self.use_huber_loss or str(getattr(self.args.training_args, "loss_func", "")).lower() == "huber"
        if use_huber:
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


