"""Frozen-PhaseFormer Stage-1 NLinear correction models.

The class deliberately lives outside :mod:`PhaseFormer`: Stage 1 must freeze a
pretrained phase model and train only the NLinear-sized correction path.  That
keeps a target-residual result from being confounded with joint adaptation of
the phase branch.
"""

from __future__ import annotations

import torch
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
