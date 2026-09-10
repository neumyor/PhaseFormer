"""Frozen-PhaseFormer Stage-1 NLinear correction models.

The class deliberately lives outside :mod:`PhaseFormer`: Stage 1 must freeze a
pretrained phase model and train only the NLinear-sized correction path.  That
keeps a target-residual result from being confounded with joint adaptation of
the phase branch.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl

from src.models.phase_adapters import WeakPeriodResidualHead


class ResidualNLinearHead(nn.Module):
    """NLinear parameterization without a level anchor, for an error target."""

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()
        return self.linear(centered).permute(0, 2, 1).contiguous()


class PooledLowRankResidualHead(nn.Module):
    """Centered NLinear correction with optional smoothing and temporal pooling.

    The final observation remains excluded from the dynamic input, matching
    NLinear's level-anchor convention. ``smooth_ratio`` blends the centered
    history with a fixed moving-average view before the pooling bottleneck.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        pool_factor: int,
        rank: int,
        smooth_ratio: float = 0.0,
        smooth_window: int = 24,
    ):
        super().__init__()
        if pool_factor < 1:
            raise ValueError("pool_factor must be >= 1")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if not 0.0 <= smooth_ratio <= 1.0:
            raise ValueError("smooth_ratio must be in [0, 1]")
        if smooth_window < 1:
            raise ValueError("smooth_window must be >= 1")
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.pool_factor = int(pool_factor)
        self.pooled_len = (self.seq_len + self.pool_factor - 1) // self.pool_factor
        if rank > min(self.pooled_len, self.pred_len):
            raise ValueError(
                f"rank={rank} exceeds factorized map limit "
                f"min({self.pooled_len}, {self.pred_len})"
            )
        self.rank = int(rank)
        self.smooth_ratio = float(smooth_ratio)
        self.smooth_window = int(smooth_window)
        self.encoder = nn.Linear(self.pooled_len, self.rank)
        self.decoder = nn.Linear(self.rank, self.pred_len)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        last = x[:, -1:, :]
        centered = (x - last).transpose(1, 2).contiguous()
        if self.smooth_ratio:
            left = (self.smooth_window - 1) // 2
            right = self.smooth_window - 1 - left
            smoothed = F.avg_pool1d(
                F.pad(centered, (left, right), mode="replicate"),
                kernel_size=self.smooth_window,
                stride=1,
            )
            centered = (
                (1.0 - self.smooth_ratio) * centered
                + self.smooth_ratio * smoothed
            )
        pooled = F.adaptive_avg_pool1d(centered, self.pooled_len)
        return self.decoder(self.encoder(pooled)).transpose(1, 2).contiguous()


class FrozenPhaseNLinearCorrection(pl.LightningModule):
    """PhaseFormer anchor plus an NLinear-sized Stage-1 correction path.

    ``mode='fusion'`` is the Stage-0 control: a full NLinear forecast is
    convexly fused with the frozen phase forecast. ``target_residual`` keeps
    that fusion but trains the NLinear-sized head on ``Y - P(X)``. ``direct``
    uses the same residual target and adds its output directly to ``P(X)``.
    """

    MODES = {"fusion", "target_residual", "direct"}

    def __init__(self, phaseformer: nn.Module, *, mode: str, learning_rate: float,
                 loss_name: str = "huber", huber_delta: float = 1.0,
                 fusion_gate_logit: torch.Tensor | None = None):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported Stage-1 mode: {mode}")
        self.phaseformer = phaseformer
        self.mode = mode
        self.learning_rate = float(learning_rate)
        self.loss_name = str(loss_name)
        self.huber_delta = float(huber_delta)
        self.pred_len = int(phaseformer.pred_len)
        self.target_var_index = int(phaseformer.target_var_index)
        head_type = WeakPeriodResidualHead if mode == "fusion" else ResidualNLinearHead
        self.correction_head = head_type(int(phaseformer.seq_len), self.pred_len)
        # The fusion gate is part of Stage-0 / Treatment-A's unchanged fusion.
        if self.mode == "fusion":
            self.fusion_gate = nn.Parameter(torch.zeros(1, 1, int(phaseformer.enc_in)))
        else:
            if self.mode == "target_residual" and fusion_gate_logit is None:
                raise ValueError("target_residual requires the frozen Stage-0 fusion gate")
            gate = torch.zeros(1, 1, int(phaseformer.enc_in)) if fusion_gate_logit is None else fusion_gate_logit
            self.register_buffer("fusion_gate", gate.detach().clone())
        self.freeze_phaseformer()

    def freeze_phaseformer(self):
        self.phaseformer.eval()
        for parameter in self.phaseformer.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen weights must also be protected from dropout/batchnorm state.
        self.phaseformer.eval()
        return self

    def _criterion(self, prediction, target):
        if self.loss_name == "huber":
            return nn.functional.huber_loss(prediction, target, delta=self.huber_delta)
        if self.loss_name == "mse":
            return nn.functional.mse_loss(prediction, target)
        if self.loss_name == "mae":
            return nn.functional.l1_loss(prediction, target)
        raise ValueError(f"unsupported loss: {self.loss_name}")

    def _phase_prediction(self, x, x_mark, y, y_mark):
        dec = self.phaseformer._build_decoder_input(y)
        with torch.no_grad():
            phase, _, _ = self.phaseformer(x, x_mark, dec, y_mark)
        return phase[:, -self.pred_len :, :]

    def _nlinear_prediction(self, x):
        # Reuse the PhaseFormer RevIN coordinate system, so the correction path
        # has the same input normalization and final-scale anchor as NLinear.
        if self.phaseformer.use_revin:
            normalized, stats = self.phaseformer.revin.normalize(x)
            branch = self.correction_head(normalized)
            if self.mode == "fusion":
                return self.phaseformer.revin.denormalize(branch, stats)
            # A residual must be rescaled but must not receive RevIN's mean.
            return branch * stats[1]
        return self.correction_head(x)

    def forward(self, x, x_mark, y, y_mark):
        phase = self._phase_prediction(x, x_mark, y, y_mark)
        branch = self._nlinear_prediction(x)
        if self.mode == "fusion":
            output = (1.0 - torch.sigmoid(self.fusion_gate)) * phase + torch.sigmoid(self.fusion_gate) * branch
            correction = output - phase
        else:
            # For target_residual/direct the NLinear head output is the learned
            # correction itself.  It retains exactly L*H+H trainable weights.
            correction = branch
            if self.mode == "target_residual":
                output = phase + torch.sigmoid(self.fusion_gate) * correction
            else:
                output = phase + correction
        return output, phase, correction

    def _target(self, batch_y):
        target = batch_y[:, -self.pred_len :, :]
        if self.target_var_index != -1:
            return target[:, :, self.target_var_index:self.target_var_index + 1]
        return target

    def _step(self, batch, split):
        x, y, x_mark, y_mark = (value.float() for value in batch)
        output, phase, correction = self(x, x_mark, y, y_mark)
        target = self._target(y)
        if self.mode == "fusion":
            loss = self._criterion(output, target)
        elif split == "train":
            loss = self._criterion(correction, target - phase)
        else:
            # Model selection is always by the deployable final forecast, even
            # though Treatment A's training target is the PhaseFormer residual.
            loss = self._criterion(output, target)
        self.log(f"{split}_loss", loss, on_epoch=True, prog_bar=split == "val")
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
        )


class FrozenPhasePooledLowRankCorrection(FrozenPhaseNLinearCorrection):
    """Frozen PhaseFormer plus a pooled, factorized direct residual correction."""

    def __init__(
        self,
        phaseformer: nn.Module,
        *,
        learning_rate: float,
        loss_name: str = "huber",
        huber_delta: float = 1.0,
        pool_factor: int,
        rank: int,
        smooth_ratio: float = 0.0,
        smooth_window: int = 24,
    ):
        super().__init__(
            phaseformer,
            mode="direct",
            learning_rate=learning_rate,
            loss_name=loss_name,
            huber_delta=huber_delta,
        )
        self.correction_head = PooledLowRankResidualHead(
            int(phaseformer.seq_len),
            int(phaseformer.pred_len),
            pool_factor=pool_factor,
            rank=rank,
            smooth_ratio=smooth_ratio,
            smooth_window=smooth_window,
        )
