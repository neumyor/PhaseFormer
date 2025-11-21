import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from types import SimpleNamespace
from typing import Optional
from gluonts.torch.util import weighted_average

from .PhaseFormer import Model

class PhaseFormerLightningModule(pl.LightningModule):
    """
    GluonTS 会用这个 LightningModule 做训练和预测。
    batch 的字段名、shape 由 Estimator 中的 InstanceSplitter 决定。
    这里我们按 GluonTS torch 系列模型的惯例：
        past_target:  (B, context_length, C)
        future_target: (B, prediction_length, C)
        past_observed_values / future_observed_values: 掩码，可选
    """

    def __init__(
        self,
        model_cfg: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ):
        super().__init__()
        # 保存超参，方便 ckpt / logger
        self.save_hyperparameters()

        # GluonTS 里的 torch 模型一般用一个 kwargs dict 来构建 model
        configs = SimpleNamespace(**model_cfg)
        self.model = Model(configs)

        # If model_cfg contains a `distr_output`, the model will return distribution outputs
        self.is_distr = getattr(configs, "distr_output", None) is not None

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(
        self, past_target: torch.Tensor, past_observed_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        给 PyTorchPredictor 用的前向接口：
        输入:  (B, context_length, C)
        输出:  (B, prediction_length, C)
        """
        # Ensure input has channel dimension: GluonTS loaders may return
        # (B, L) for univariate series. PhaseFormer.Model expects (B, L, C).
        if past_target.dim() == 2:
            past_target = past_target.unsqueeze(-1)
        out = self.model(past_target)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def _ensure_channel_dim(x: torch.Tensor) -> torch.Tensor:
        """
        若输入为 (B, L)，则补成 (B, L, 1)；否则原样返回。
        """
        if x.dim() == 2:
            return x.unsqueeze(-1)
        return x

    def _compute_deterministic_loss(
        self,
        past_target: torch.Tensor,           # (B, L_in, C)
        future_target: torch.Tensor,         # (B, L_out, C)
        future_observed_values: torch.Tensor # (B, L_out, C)
    ) -> torch.Tensor:
        """
        确定性分支的 MSE 损失，按通道做加权平均（与 probabilistic 分支保持同一风格）。
        """
        # 预测
        y_hat = self.forward(past_target)    # (B, L_out, C)

        # 元素级 MSE
        loss_raw = (y_hat - future_target) ** 2   # (B, L_out, C)

        # 和官方 probabilistic 一样，在通道维 dim=-1 上用 future_observed_values 做加权平均
        # 得到 (B, L_out) 的 loss
        loss_per_ts = weighted_average(
            loss_raw,
            weights=future_observed_values,
            dim=-1,
        )  # (B, L_out)

        # 再对 batch 和时间维做平均，得到标量
        loss = loss_per_ts.mean()
        return loss

    def _compute_probabilistic_loss(
        self,
        past_target: torch.Tensor,           # (B, L_in, C)
        past_observed_values: torch.Tensor,  # (B, L_in, C)
        future_target: torch.Tensor,         # (B, L_out, C)
        future_observed_values: torch.Tensor # (B, L_out, C)
    ) -> torch.Tensor:
        """
        概率分支的 NLL 损失，严格对齐官方逻辑：
        1) 调用 forward 得到 distr_args, loc, scale
        2) self.model.distr_output.loss(target, distr_args, loc, scale)
        3) weighted_average(loss, weights=future_observed_values, dim=-1)
        4) 对 (B, T) 取平均
        """
        # 得到分布参数；forward 的签名和官方一致
        distr_args, loc, scale = self.forward(
            past_target=past_target,
            past_observed_values=past_observed_values,
        )
        # distr_output.loss 内部会根据 distr_args 形状自己处理 tuple / tensor 等情况
        # 返回形状通常为 (B, L_out, C)
        raw_loss = self.model.distr_output.loss(
            target=future_target,
            distr_args=distr_args,
            loc=loc,
            scale=scale,
        )  # (B, L_out, C)

        # 官方逻辑：在通道维上用 future_observed_values 做加权平均
        loss_per_ts = weighted_average(
            raw_loss,
            weights=future_observed_values,
            dim=-1,
        )  # (B, L_out)

        # 再对 batch 和时间维做平均，得到标量
        loss = loss_per_ts.mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        past_target = batch["past_target"]          # (B, L_in, C) or (B, L_in)
        future_target = batch["future_target"]      # (B, L_out, C) or (B, L_out)
        past_observed_values = batch.get(
            "past_observed_values", torch.ones_like(past_target)
        )
        future_observed_values = batch.get(
            "future_observed_values", torch.ones_like(future_target)
        )

        # 统一补齐 channel 维度
        past_target = self._ensure_channel_dim(past_target)
        future_target = self._ensure_channel_dim(future_target)
        past_observed_values = self._ensure_channel_dim(past_observed_values)
        future_observed_values = self._ensure_channel_dim(future_observed_values)

        # 根据是否为分布输出选择不同 loss 计算方式
        if not self.is_distr:
            loss = self._compute_deterministic_loss(
                past_target=past_target,
                future_target=future_target,
                future_observed_values=future_observed_values,
            )
        else:
            loss = self._compute_probabilistic_loss(
                past_target=past_target,
                past_observed_values=past_observed_values,
                future_target=future_target,
                future_observed_values=future_observed_values,
            )

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        past_target = batch["past_target"]          # (B, L_in, C) or (B, L_in)
        future_target = batch["future_target"]      # (B, L_out, C) or (B, L_out)
        past_observed_values = batch.get(
            "past_observed_values", torch.ones_like(past_target)
        )
        future_observed_values = batch.get(
            "future_observed_values", torch.ones_like(future_target)
        )

        # 同样补齐 channel 维度
        past_target = self._ensure_channel_dim(past_target)
        future_target = self._ensure_channel_dim(future_target)
        past_observed_values = self._ensure_channel_dim(past_observed_values)
        future_observed_values = self._ensure_channel_dim(future_observed_values)

        if not self.is_distr:
            loss = self._compute_deterministic_loss(
                past_target=past_target,
                future_target=future_target,
                future_observed_values=future_observed_values,
            )
        else:
            loss = self._compute_probabilistic_loss(
                past_target=past_target,
                past_observed_values=past_observed_values,
                future_target=future_target,
                future_observed_values=future_observed_values,
            )

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss