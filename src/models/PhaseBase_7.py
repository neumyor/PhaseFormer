import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.pl_bases.default_module import DefaultPLModule
from src.models.layers.SelfAttention_Family import FullAttention, AttentionLayer


class RevIN(nn.Module):
    """
    Reversible Instance Normalization over time (per-sample, per-variable).
    对输入 x: (B, L, C) 在时间轴 L 做标准化；并提供反归一化。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):  # x: (B, L, C)
        mu = x.mean(dim=1, keepdim=True)  # (B,1,C)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        xn = (x - mu) / sigma
        if self.affine:
            xn = xn * self.weight + self.bias
        return xn, (mu, sigma)

    def denormalize(self, y, stats):  # y: (B, L', C)
        mu, sigma = stats
        return y * sigma + mu


class DimensionReductionAttention(nn.Module):
    """
    降维注意力模块：参考CAPEN.py，使用固定数量的router降低attention复杂度。
    核心：router聚合 → router分发 → 残差+Norm → MLP → 残差+Norm。
    输入/输出：Z: (B, C, L, D)
    """

    def __init__(
        self,
        latent_dim: int,
        num_routers: int = 8,
        period_len: int = 24,
        **kwargs,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_routers = num_routers
        self.period2basis = nn.Linear(period_len, num_routers)
        self.basis2period = nn.Linear(num_routers, period_len)

    def forward(self, Z):  # Z: (B, C, L, D)
        B, C, L, D = Z.shape
        x = Z.permute(0, 1, 3, 2)
        x = self.period2basis(x)
        x = self.basis2period(x)
        x = x.permute(0, 1, 3, 2)
        return x


class PhaseSeriesEncoder(nn.Module):
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
    def __init__(
        self,
        latent_dim: int,
        out_steps: int,
        hidden: int = 64,
        use_mlp: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_steps = out_steps
        self.norm = nn.LayerNorm(latent_dim)
        if use_mlp:
            self.projection = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_steps),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(latent_dim, out_steps),
            )

    def forward(self, Z):  # (B, C, L, D)
        Zn = self.norm(Z)
        y = self.projection(Zn)
        return y


class PhaseBase(DefaultPLModule):
    """
    PhaseBase：严格依据 CAPEN.py 的降维注意力版本（只保留 DimReduction）。
    流程：RevIN → PhaseSeriesEncoder → DimensionReductionAttention → PhasePredictor。
    """

    def __init__(self, configs):
        super().__init__(configs)

        # 基础参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        # 维度与结构
        self.latent_dim = getattr(configs, "latent_dim", 8)
        self.phase_encoder_hidden = getattr(configs, "phase_encoder_hidden", 32)
        self.predictor_hidden = getattr(configs, "predictor_hidden", 64)

        # 降维注意力参数（与 CAPEN.py 一致的命名）
        self.phase_attn_heads = getattr(configs, "phase_attn_heads", 4)
        self.phase_attn_dropout = getattr(configs, "phase_attn_dropout", 0.0)
        self.phase_attn_use_relpos = getattr(configs, "phase_attn_use_relpos", True)
        self.phase_attn_window = getattr(configs, "phase_attn_window", None)
        self.phase_attention_dim = getattr(configs, "phase_attention_dim", None)
        self.phase_num_routers = getattr(configs, "phase_num_routers", 8)
        self.phase_use_pos_embed = getattr(configs, "phase_use_pos_embed", False)
        self.phase_pos_dropout = getattr(configs, "phase_pos_dropout", 0.0)

        # 计算周期数（输入/输出）
        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (
            self.pred_len + self.period_len - 1
        ) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        # 归一化相关
        self.use_revin = getattr(configs, "use_revin", True)
        self.revin_affine = getattr(configs, "revin_affine", False)
        self.revin_eps = getattr(configs, "revin_eps", 1e-5)
        if self.use_revin:
            self.revin = RevIN(
                num_features=self.enc_in, eps=self.revin_eps, affine=self.revin_affine
            )

        # 模块组装
        self.phase_encoder = PhaseSeriesEncoder(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim,
            hidden=self.phase_encoder_hidden,
            use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
            dropout=getattr(configs, "phase_encoder_dropout", 0.0),
        )

        self.predictor = PhasePredictor(
            latent_dim=self.latent_dim,
            out_steps=self.num_periods_output,
            hidden=self.predictor_hidden,
            use_mlp=getattr(configs, "predictor_use_mlp", False),
            dropout=getattr(configs, "predictor_dropout", 0.0),
        )

    # --------- 工具：相位重排 ----------
    @staticmethod
    def _to_phase_series(x_periods):
        """(B, C, P_in, L) -> (B, C, L, P_in)"""
        return x_periods.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _from_phase_steps_to_periods(y_phase_steps):
        """(B, C, L, P_out) -> (B, C, P_out, L)"""
        return y_phase_steps.permute(0, 1, 3, 2).contiguous()

    # ------------- 前向 --------------
    def forward(
        self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs
    ):
        """
        输入: x_enc (B, seq_len, C)
        输出: y_hat (B, pred_len, C), Z_encoded (B,C,L,D), future_phase_vals (B,C,L,P_out)
        """
        # 1) RevIN
        if self.use_revin:
            x_in, stats = self.revin.normalize(x_enc.float())  # (B, L, C)
        else:
            x_in, stats = x_enc.float(), None

        x = x_in.permute(0, 2, 1)  # (B, C, L_total)
        B, C, L_total = x.shape

        # 2) 环形补齐到整 period
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode="circular")  # (B, C, total_len_in)

        # 3) 切成 (B, C, P_in, L)
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)

        # 4) 相位并行：按相位取列 (B,C,L,P_in)
        phase_series = self._to_phase_series(x_periods)  # (B,C,L,P_in)

        # 5) 相位编码 -> (B,C,L,D)
        Z = self.phase_encoder(phase_series)

        # 6) 跳过相位轴交互

        # 7) 逐相位预测 P_out 步
        y_phase_steps = self.predictor(Z)  # (B,C,L,P_out)

        # 8) 还原为 (B,C,P_out,L) 并拼为序列
        y_periods = self._from_phase_steps_to_periods(y_phase_steps)  # (B,C,P_out,L)
        y_full = y_periods.reshape(B, C, -1)[..., : self.pred_len]  # (B,C,pred_len)
        y_hat = y_full.permute(0, 2, 1)  # (B,pred_len,C)

        # 9) 反归一化回原标度
        if stats is not None:
            y_hat = self.revin.denormalize(y_hat, stats)

        return y_hat, Z, y_phase_steps

    # --------- Lightning 步骤 ----------
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        # 工程兼容：仍然构造 dec_inp（虽未使用）
        dec_inp = self._build_decoder_input(batch_y)

        outputs, Z, _ = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        main_loss = criterion(outputs, target)
        loss = main_loss

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
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        loss = criterion(outputs, target)
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
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        from src.utils.metrics import metric

        m = metric(outputs.detach(), target.detach())
        self.log_dict({f"test_{k}": v for k, v in m.items()}, on_epoch=True)
        return m
