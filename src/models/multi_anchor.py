"""Out-of-fold router over complete, frozen PhaseFormer anchors.

The router is deliberately small.  It observes encoder history and descriptors
of forecasts that are already available at inference, then chooses A1, I0 or
R0 independently for each sample, channel and future cycle.  It never receives
the target or decoder-side future values.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pl_bases.default_module import DefaultPLModule


ANCHOR_NAMES = ("A1", "I0", "R0")


class MultiAnchorRouter(nn.Module):
    """Route three complete forecasts at future-cycle granularity."""

    MODES = {"global", "structural"}
    OUTPUT_MODES = {"hard", "soft"}

    def __init__(
        self,
        pred_len: int,
        period_len: int = 24,
        mode: str = "structural",
        output_mode: str = "hard",
        hidden: int = 24,
        temperature: float = 0.2,
        eps: float = 1e-6,
        soft_a1_prior: float = 0.98,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported router mode: {mode}")
        if output_mode not in self.OUTPUT_MODES:
            raise ValueError(f"unsupported output mode: {output_mode}")
        if pred_len <= 0 or period_len <= 1:
            raise ValueError("pred_len must be positive and period_len > 1")
        if pred_len % period_len:
            raise ValueError("multi-anchor routing requires whole future cycles")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.pred_len = int(pred_len)
        self.period_len = int(period_len)
        self.num_future_cycles = self.pred_len // self.period_len
        self.mode = mode
        self.output_mode = output_mode
        self.temperature = float(temperature)
        self.eps = float(eps)

        if output_mode == "soft":
            if not 1.0 / 3.0 < soft_a1_prior < 1.0:
                raise ValueError("soft_a1_prior must be in (1/3, 1)")
            other = (1.0 - soft_a1_prior) / 2.0
            # Account for the routing temperature so the actual initial
            # softmax probabilities, rather than the raw logits, equal the
            # documented prior.
            initial = self.temperature * torch.tensor(
                (soft_a1_prior, other, other)
            ).log()
        else:
            # A tied logit vector has deterministic argmax A1, so the hard
            # forward pass is bit-exactly the complete A1 forecast at init.
            initial = torch.zeros(3)
        self.global_logits = nn.Parameter(initial)

        if mode == "structural":
            # 4 history descriptors + 4 descriptors for each of three anchors.
            self.router = nn.Sequential(
                nn.Linear(16, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3),
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
            self.future_cycle_bias = nn.Parameter(
                torch.zeros(self.num_future_cycles, 3)
            )

        self.last_logits = None
        self.last_soft_weights = None
        self.last_hard_weights = None
        self.last_weights = None
        self.last_features = None

    def _history_features(self, history: torch.Tensor):
        if history.ndim != 3:
            raise ValueError("history must have shape (B,L,C)")
        if history.shape[1] < 2 * self.period_len:
            raise ValueError("history must contain at least two periods")
        x = history.float()
        scale = x.std(dim=1, unbiased=False).clamp_min(self.eps)
        current = x[:, -self.period_len :, :]
        previous = x[:, -2 * self.period_len : -self.period_len, :]
        drift = (current.mean(dim=1) - previous.mean(dim=1)) / scale

        left = x[:, :-self.period_len, :]
        right = x[:, self.period_len :, :]
        left = left - left.mean(dim=1, keepdim=True)
        right = right - right.mean(dim=1, keepdim=True)
        correlation = (left * right).mean(dim=1) / (
            left.square().mean(dim=1).sqrt()
            * right.square().mean(dim=1).sqrt()
        ).clamp_min(self.eps)
        correlation = correlation.clamp(-1.0, 1.0)

        diff_volatility = (x[:, 1:, :] - x[:, :-1, :]).std(
            dim=1, unbiased=False
        ) / scale
        cycle_count = x.shape[1] // self.period_len
        cycles = x[:, -cycle_count * self.period_len :, :].view(
            x.shape[0], cycle_count, self.period_len, x.shape[2]
        )
        slot_signal = cycles.mean(dim=1).var(dim=1, unbiased=False)
        total = cycles.var(dim=(1, 2), unbiased=False)
        phase_reliability = slot_signal / (total + self.eps)
        features = torch.stack(
            (drift, correlation, diff_volatility, phase_reliability), dim=-1
        )
        return features.clamp(-10.0, 10.0), scale

    def _forecast_features(
        self, history: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        if anchors.ndim != 4 or anchors.shape[-1] != 3:
            raise ValueError("anchors must have shape (B,H,C,3)")
        if anchors.shape[1] != self.pred_len:
            raise ValueError("anchor horizon does not match router horizon")
        history_features, scale = self._history_features(history)
        B, _, C, _ = anchors.shape
        cycles = anchors.view(
            B, self.num_future_cycles, self.period_len, C, 3
        ).permute(0, 3, 1, 2, 4)
        local_scale = scale[:, :, None, None].clamp_min(self.eps)
        last = history[:, -1, :].float()[:, :, None, None]
        displacement = (cycles.mean(dim=3) - last) / local_scale
        roughness = (cycles[:, :, :, 1:, :] - cycles[:, :, :, :-1, :]).abs().mean(
            dim=3
        ) / local_scale
        consensus = cycles.mean(dim=-1, keepdim=True)
        disagreement = (cycles - consensus).abs().mean(dim=3) / local_scale
        slope = (cycles[:, :, :, -1, :] - cycles[:, :, :, 0, :]) / local_scale
        # (B,C,Q,3 anchors,4 features) -> (B,C,Q,12)
        anchor_features = torch.stack(
            (displacement, roughness, disagreement, slope), dim=-1
        ).reshape(B, C, self.num_future_cycles, 12)
        history_features = history_features.unsqueeze(2).expand(
            -1, -1, self.num_future_cycles, -1
        )
        return torch.cat((history_features, anchor_features), dim=-1).clamp(
            -10.0, 10.0
        )

    def cycle_logits(self, history: torch.Tensor, anchors: torch.Tensor):
        B, _, C, _ = anchors.shape
        base = self.global_logits.view(1, 1, 1, 3).expand(
            B, C, self.num_future_cycles, 3
        )
        if self.mode == "global":
            logits = base
            features = None
        else:
            features = self._forecast_features(history, anchors)
            logits = (
                base
                + self.router(features)
                + self.future_cycle_bias.view(1, 1, self.num_future_cycles, 3)
            )
        self.last_logits = logits
        self.last_features = None if features is None else features.detach()
        return logits

    def forward(self, history: torch.Tensor, *anchor_forecasts: torch.Tensor):
        if len(anchor_forecasts) != 3:
            raise ValueError("exactly three anchor forecasts are required")
        anchors = torch.stack(anchor_forecasts, dim=-1)
        logits = self.cycle_logits(history, anchors)
        soft = torch.softmax(logits / self.temperature, dim=-1)
        indices = logits.argmax(dim=-1)
        hard = F.one_hot(indices, num_classes=3).to(dtype=soft.dtype)
        if self.output_mode == "hard":
            weights = hard.detach() - soft.detach() + soft if self.training else hard
        else:
            weights = soft
        point_weights = weights.repeat_interleave(self.period_len, dim=2).permute(
            0, 2, 1, 3
        )
        output = (anchors * point_weights).sum(dim=-1)
        self.last_soft_weights = soft.detach()
        self.last_hard_weights = hard.detach()
        self.last_weights = point_weights.detach()
        return output, point_weights


class MultiAnchorPhaseFormer(DefaultPLModule):
    """Lightning wrapper that trains only a router over frozen anchors."""

    def __init__(
        self,
        configs,
        shadow_anchors: dict[str, nn.Module],
        full_anchors: dict[str, nn.Module],
        *,
        router_mode: str = "structural",
        output_mode: str = "hard",
        hidden: int = 24,
        temperature: float = 0.2,
        oracle_temperature: float = 0.1,
        route_weight: float = 0.1,
        mean_regret_weight: float = 0.0,
        cvar_weight: float = 0.0,
    ):
        super().__init__(configs)
        if tuple(shadow_anchors) != ANCHOR_NAMES or tuple(full_anchors) != ANCHOR_NAMES:
            raise ValueError(f"anchor dictionaries must be ordered {ANCHOR_NAMES}")
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.period_len = int(configs.period_len)
        self.shadow_anchors = nn.ModuleDict(shadow_anchors)
        self.full_anchors = nn.ModuleDict(full_anchors)
        self.router = MultiAnchorRouter(
            self.pred_len,
            self.period_len,
            mode=router_mode,
            output_mode=output_mode,
            hidden=hidden,
            temperature=temperature,
        )
        self.oracle_temperature = float(oracle_temperature)
        self.route_weight = float(route_weight)
        self.mean_regret_weight = float(mean_regret_weight)
        self.cvar_weight = float(cvar_weight)
        self.output_mode = output_mode
        self.router_mode = router_mode
        self.last_anchor_outputs = None
        self.last_anchor_source = None
        self.last_oracle = None
        self.last_relative_regret = None
        self.freeze_anchors()

    def freeze_anchors(self):
        for bank in (self.shadow_anchors, self.full_anchors):
            bank.eval()
            for parameter in bank.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen forecasts must be deterministic even while the wrapper trains.
        self.shadow_anchors.eval()
        self.full_anchors.eval()
        self.router.train(mode)
        return self

    def on_train_epoch_start(self):
        self.freeze_anchors()
        self.router.train()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.router.parameters(), lr=self.args.training_args.learning_rate
        )
        if self.args.training_args.lr_schedule_config.type == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.training_args.lr_schedule_config.tmax,
                eta_min=1e-8,
            )
            return [optimizer], [scheduler]
        return optimizer

    @staticmethod
    def _anchor_output(model, x_enc, x_mark_enc, x_dec, x_mark_dec):
        value = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return value[0] if isinstance(value, tuple) else value

    def forward(
        self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs
    ):
        bank = self.shadow_anchors if self.training else self.full_anchors
        self.last_anchor_source = "shadow" if self.training else "full"
        with torch.no_grad():
            forecasts = tuple(
                self._anchor_output(bank[name], x_enc, x_mark_enc, x_dec, x_mark_dec)
                for name in ANCHOR_NAMES
            )
        self.last_anchor_outputs = forecasts
        output, _ = self.router(x_enc, *forecasts)
        return output, None, None

    def _cyclewise_mse(self, prediction: torch.Tensor, target: torch.Tensor):
        B, H, C = prediction.shape
        return (prediction - target).square().view(
            B, self.pred_len // self.period_len, self.period_len, C
        ).mean(dim=2).permute(0, 2, 1)

    def _routing_losses(self, output: torch.Tensor, target: torch.Tensor):
        anchor_mse = torch.stack(
            [self._cyclewise_mse(value, target) for value in self.last_anchor_outputs],
            dim=-1,
        )
        envelope = anchor_mse.min(dim=-1).values
        relative_anchor_error = (
            (anchor_mse - envelope.unsqueeze(-1))
            / envelope.unsqueeze(-1).clamp_min(1e-8)
        )
        oracle = torch.softmax(
            -relative_anchor_error.detach() / self.oracle_temperature, dim=-1
        )
        route = -(
            oracle
            * F.log_softmax(
                self.router.last_logits / self.router.temperature, dim=-1
            )
        ).sum(dim=-1).mean()
        candidate_mse = self._cyclewise_mse(output, target)
        relative_regret = (
            candidate_mse - envelope
        ) / envelope.clamp_min(1e-8)
        positive = F.relu(relative_regret)
        mean_regret = positive.mean()
        flat = positive.reshape(-1)
        tail_count = max(1, math.ceil(0.1 * flat.numel()))
        cvar = flat.topk(tail_count).values.mean()
        self.last_oracle = oracle.detach()
        self.last_relative_regret = relative_regret.detach()
        return route, mean_regret, cvar

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        dec_inp = self._build_decoder_input(batch_y)
        outputs, _, _ = self(
            batch_x, batch_x_mark.float(), dec_inp, batch_y_mark.float()
        )
        target = batch_y[:, -self.pred_len :, :]
        outputs = outputs[:, -self.pred_len :, :]
        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index : self.target_var_index + 1]
            outputs = outputs[:, :, self.target_var_index : self.target_var_index + 1]
            self.last_anchor_outputs = tuple(
                value[:, :, self.target_var_index : self.target_var_index + 1]
                for value in self.last_anchor_outputs
            )
        prediction_loss = self._get_criterion(
            self.args.training_args.loss_func
        )(outputs, target)
        route, mean_regret, cvar = self._routing_losses(outputs, target)
        loss = (
            prediction_loss
            + self.route_weight * route
            + self.mean_regret_weight * mean_regret
            + self.cvar_weight * cvar
        )
        self.log_dict(
            {
                "train_loss": loss,
                "train_prediction": prediction_loss,
                "train_route": route,
                "train_regret": mean_regret,
                "train_cvar": cvar,
            },
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_y = batch_y.float()
        outputs, _, _ = self(
            batch_x.float(),
            batch_x_mark.float(),
            self._build_decoder_input(batch_y),
            batch_y_mark.float(),
        )
        target = batch_y[:, -self.pred_len :, :]
        outputs = outputs[:, -self.pred_len :, :]
        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index : self.target_var_index + 1]
            outputs = outputs[:, :, self.target_var_index : self.target_var_index + 1]
        loss = self._get_criterion(self.args.training_args.loss_func)(outputs, target)
        self.log("val_loss", loss, on_epoch=True)
        return loss
