import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pl_bases.default_module import DefaultPLModule


# -----------------------------
# 相位轴上的轻量环形深度可分卷积
# -----------------------------
class DepthwiseCircularConv1dPhase(nn.Module):
    """
    在相位轴 L 上做 depthwise 1D 卷积，但仅按潜维 D 建组（groups = latent_dim）。
    做法：把变量 C 并到 batch 维 (B*C, D, L)，使同一套 D 个核在所有变量上共享。
    """
    def __init__(self, channels: int, latent_dim: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size 建议奇数"
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        # 每个潜维一个核，跨变量共享
        self.dw = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=latent_dim,
            kernel_size=kernel_size,
            padding=0,
            groups=latent_dim,
            bias=True,
        )

    def forward(self, Z):  # Z: (B, C, L, D)
        B, C, L, D = Z.shape
        # (B, C, L, D) -> (B*C, D, L)
        x = Z.permute(0, 1, 3, 2).reshape(B * C, D, L)
        pad = self.kernel_size - 1
        x = F.pad(x, (pad, 0), mode="circular")   # 仅在左侧做环形补齐，保持中心对齐
        x = self.dw(x)                            # (B*C, D, L)
        # (B*C, D, L) -> (B, C, L, D)
        x = x.view(B, C, D, L).permute(0, 1, 3, 2)
        return x

# -----------------------------
# 相位序列编码器
# -----------------------------
class PhaseSeriesEncoder(nn.Module):
    """
    对“每个相位的跨周期序列（长度 P_in）”做共享编码 → latent_dim
    """
    def __init__(self, p_in: int, latent_dim: int):
        super().__init__()
        self.projection = nn.Linear(p_in, latent_dim)

    def forward(self, phase_series):  # (B, C, L, P_in)
        return self.projection(phase_series)


# -----------------------------
# 逐相位预测器：latent -> 多步（P_out）未来相位值
# -----------------------------
class PhasePredictor(nn.Module):
    """
    对每个相位的 latent 做映射直接输出 P_out 步的未来值。
    """
    def __init__(self, latent_dim: int, out_steps: int, hidden: int = 64):
        super().__init__()
        self.out_steps = out_steps
        self.norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, out_steps),
        )

    def forward(self, Z):  # (B, C, L, D)
        Zn = self.norm(Z)
        y = self.ffn(Zn) # (B, C, L, P_out)
        return y



class CAPE(DefaultPLModule):
    """
    CAPE（Circular Autoencoder for Periodic Equivariant forecasting, phase-parallel）
    —— 相位并行：每个相位跨周期编码 -> 预测该相位未来 P_out 步 -> 拼成未来序列
    """
    def __init__(self, configs):
        super().__init__(configs)

        # 基础参数（与工程规范保持一致）
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len  # L

        # 维度与结构超参（命名兼容：若没给专用字段，则回退到常用字段）
        self.latent_dim = getattr(configs, "latent_dim", 8)

        # 编码器隐藏维（兼容 encoder_hidden_dims / phase_encoder_hidden）
        _enc_hid_list = getattr(configs, "encoder_hidden_dims", None)
        self.phase_encoder_hidden = getattr(
            configs, "phase_encoder_hidden",
            (_enc_hid_list[0] if isinstance(_enc_hid_list, (list, tuple)) and len(_enc_hid_list) > 0 else 32)
        )

        # 相位交互卷积开关与 kernel
        self.use_phase_interaction = getattr(configs, "use_phase_interaction", True)
        self.phase_kernel_size = getattr(configs, "phase_kernel_size", 3)

        # 预测器隐藏维（兼容 predictor_hidden）
        self.predictor_hidden = getattr(configs, "predictor_hidden", 64)

        # 等变正则（默认关闭，保持 simple）
        self.use_equivariance_regularization = getattr(configs, "use_equivariance_regularization", False)
        self.equivariance_weight = getattr(configs, "equivariance_weight", 0.1)
        self.num_shifts_reg = getattr(configs, "num_shifts_reg", 2)

        # 计算周期数（输入/输出）
        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        # 模块组装
        self.phase_encoder = PhaseSeriesEncoder(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim
        )
        if self.use_phase_interaction:
            self.phase_interact = DepthwiseCircularConv1dPhase(
                channels=self.enc_in, latent_dim=self.latent_dim, kernel_size=self.phase_kernel_size
            )
        else:
            self.phase_interact = nn.Identity()

        self.predictor = PhasePredictor(
            latent_dim=self.latent_dim,
            out_steps=self.num_periods_output,
            hidden=self.predictor_hidden
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
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        输入: x_enc (B, seq_len, C)
        输出: y_hat (B, pred_len, C), Z_encoded (B,C,L,D), future_phase_vals (B,C,L,P_out)
        """
        x = x_enc.float().permute(0, 2, 1)  # (B, C, L_total)
        B, C, L_total = x.shape

        # 补齐到整 period（环形）
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode='circular')  # (B, C, total_len_in)

        # 切成 (B, C, P_in, L)
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)

        # 相位并行：按相位取列 (B,C,L,P_in) → 编码 (B,C,L,D)
        phase_series = self._to_phase_series(x_periods)
        Z = self.phase_encoder(phase_series)

        # 相位轴交互
        Z = self.phase_interact(Z)  # (B,C,L,D)

        # 逐相位预测 P_out 步未来值 (B,C,L,P_out)
        y_phase_steps = self.predictor(Z)

        # 还原为 (B,C,P_out,L) 并拼为序列
        y_periods = self._from_phase_steps_to_periods(y_phase_steps)         # (B,C,P_out,L)
        y_full = y_periods.reshape(B, C, -1)[..., :self.pred_len]            # (B,C,pred_len)
        y_hat = y_full.permute(0, 2, 1)                                      # (B,pred_len,C)
        return y_hat, Z, y_phase_steps

    # --------- Lightning 步骤 ----------
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        # 与工程兼容：仍然构造 dec_inp（虽未使用）
        dec_inp = self._build_decoder_input(batch_y)

        outputs, Z, _ = self(
            x_enc=batch_x, x_mark_enc=batch_x_mark, x_dec=dec_inp, x_mark_dec=batch_y_mark
        )

        outputs = outputs[:, -self.pred_len:, :]
        target = batch_y[:, -self.pred_len:, :]

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
            x_enc=batch_x, x_mark_enc=batch_x_mark, x_dec=dec_inp, x_mark_dec=batch_y_mark
        )

        outputs = outputs[:, -self.pred_len:, :]
        target = batch_y[:, -self.pred_len:, :]

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
            x_enc=batch_x, x_mark_enc=batch_x_mark, x_dec=dec_inp, x_mark_dec=batch_y_mark
        )

        outputs = outputs[:, -self.pred_len:, :]
        target = batch_y[:, -self.pred_len:, :]

        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        from src.utils.metrics import metric
        m = metric(outputs.detach(), target.detach())
        self.log_dict({f"test_{k}": v for k, v in m.items()}, on_epoch=True)
        return m