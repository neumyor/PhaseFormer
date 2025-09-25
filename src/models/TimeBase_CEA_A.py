import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pl_bases.default_module import DefaultPLModule


# -----------------------------
# 基础构件：环形卷积与环形工具
# -----------------------------
class CircularConv1d(nn.Module):
    """带环形填充的一维卷积（任意kernel/dilation）"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, x):
        # x: (B, C, L)
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0), mode='circular')
        return self.conv(x)


def circular_interpolate(x, target_len, mode="linear"):
    """将 (B, C, L) 通过插值放缩到 target_len，前后视作环"""
    B, C, L = x.shape
    if L == target_len:
        return x
    # 拼接首元素到末尾，减小端点处插值的不连续
    x_ext = torch.cat([x, x[..., :1]], dim=-1)  # (B,C,L+1)
    x_ext = F.interpolate(x_ext, size=target_len+1, mode=mode, align_corners=False if mode=="linear" else None)
    return x_ext[..., :target_len]


# -----------------------------
# 编码器 / 解码器（长度安全 + 环形）
# -----------------------------
class CircularEquivariantEncoder(nn.Module):
    """
    环先验编码器：仅用环形卷积与局部特征聚合；输出固定维的 latent 向量
    通过 AdaptiveAvgPool 压缩到 L_enc；随后线性压平到 latent_dim
    """
    def __init__(self, period_len, latent_dim, hidden_dims=None, enc_ratio=4):
        super().__init__()
        self.period_len = period_len
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [32, 64]
        self.enc_ratio = max(2, enc_ratio)  # 至少压缩到 L/2，稳妥

        L_enc = max(1, round(period_len / self.enc_ratio))
        self.L_enc = L_enc

        self.trunk = nn.Sequential(
            CircularConv1d(1, self.hidden_dims[0], kernel_size=3),
            nn.GELU(),
            CircularConv1d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(L_enc),
        )
        self.fc = nn.Linear(self.hidden_dims[1] * L_enc, latent_dim)

    def forward(self, x):
        # x: (B, L) or (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.trunk(x)                          # (B, H2, L_enc)
        h = h.flatten(1)                           # (B, H2*L_enc)
        z = self.fc(h)                             # (B, D)
        return z


class CircularEquivariantDecoder(nn.Module):
    """
    长度安全的解码器：先展开到 (H, L_enc)，再用插值精确对齐到 period_len，
    最后用环形卷积细化重建，避免 ConvTranspose 产生的边界与整除约束。
    """
    def __init__(self, latent_dim, period_len, hidden_dims=None, L_enc=None):
        super().__init__()
        self.period_len = period_len
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        assert L_enc is not None, "Decoder 需要知道编码端的 L_enc 以对齐形状"
        self.L_enc = L_enc

        self.expand = nn.Linear(latent_dim, self.hidden_dims[0] * self.L_enc)
        self.refine = nn.Sequential(
            CircularConv1d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=3),
            nn.GELU(),
            CircularConv1d(self.hidden_dims[1], self.hidden_dims[1], kernel_size=3),
            nn.GELU(),
            CircularConv1d(self.hidden_dims[1], 1, kernel_size=3),
        )

    def forward(self, z):
        # z: (B, D)
        x = self.expand(z)                         # (B, H0*L_enc)
        x = x.view(z.size(0), self.hidden_dims[0], self.L_enc)
        x = circular_interpolate(x, self.period_len, mode="linear")  # (B, H0, L)
        x = self.refine(x)                         # (B, 1, L)
        return x.squeeze(1)                        # (B, L)



class PeriodwiseResMLP(nn.Module):
    """
    在 period 维（长度 P_in -> P_out）做映射：
      - 先线性到隐藏维，再GELU，再线性到 P_out
      - 残差/LayerNorm 稳定训练
      - 对每个 latent 维度与变量共享参数（shared across (enc_in, latent_dim)）
    """
    def __init__(self, num_periods_input, num_periods_output, hidden=128, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(num_periods_input, hidden)
        self.lin2 = nn.Linear(hidden, num_periods_output)
        self.norm = nn.LayerNorm(num_periods_output)
        self.drop = nn.Dropout(dropout)
        # 旁路：直接线性相加（类似ResNet）
        self.skip = nn.Linear(num_periods_input, num_periods_output)

    def forward(self, x):
        """
        x: (B, enc_in, P_in, D)  —— 对 D 与 enc_in 参数共享
        out: (B, enc_in, P_out, D)
        """
        B, C, P_in, D = x.shape
        x_ = x.permute(0, 1, 3, 2).reshape(B*C*D, P_in)  # (B*C*D, P_in)

        y = self.lin2(F.gelu(self.lin1(x_)))
        y = self.drop(y) + self.skip(x_)
        y = self.norm(y)

        P_out = y.size(-1)
        y = y.view(B, C, D, P_out).permute(0, 1, 3, 2)   # (B, C, P_out, D)
        return y


# -----------------------------
# 等变/不变 正则（向量化稳定版）
# -----------------------------
class EquivarianceRegularizer(nn.Module):
    """
    mode = "invariant":   E(x) ≈ E(roll(x,k))          —— 不变性
    mode = "permute":     E(roll(x,k)) ≈ P(k)E(x)      —— 等变（需定义 P）
    """
    def __init__(self, mode="invariant", num_shifts=3):
        super().__init__()
        assert mode in ["invariant", "permute"]
        self.mode = mode
        self.num_shifts = num_shifts

    def forward(self, encoder, x_periods, latent_dim, device=None):
        """
        x_periods: (B, enc_in, P_in, L)
        encoder:  期级编码器（单 period -> latent）
        """
        B, C, P, L = x_periods.shape
        device = device or x_periods.device

        # 采样若干个 k（避免 k=0）
        ks = torch.randint(1, max(2, L), size=(self.num_shifts,), device=device)

        # 展平：一次性编码
        x_flat = x_periods.reshape(B*C*P, L)  # (B*C*P, L)
        z = encoder(x_flat)                   # (B*C*P, D)

        losses = []
        for k in ks:
            xr = torch.roll(x_periods, shifts=int(k.item()), dims=-1).reshape(B*C*P, L)
            zr = encoder(xr)

            if self.mode == "invariant":
                losses.append(F.mse_loss(zr, z))
            else:
                # 等变占位：定义一个对 latent 的“循环置换” P(k)
                # 你可以替换成任意已知群作用在 latent 上的线性/置换算子
                Pk = torch.roll(torch.eye(latent_dim, device=device), shifts=int(k.item()) % latent_dim, dims=0)
                z_perm = z @ Pk  # (B*C*P, D)
                losses.append(F.mse_loss(zr, z_perm))

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


# -----------------------------
# 主模型
# -----------------------------
class CAPE(DefaultPLModule):
    """TimeBase 循环等变自编码器（修正版）"""
    def __init__(self, configs):
        super().__init__(configs)
        # 基础参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        # 模型维度
        self.latent_dim = getattr(configs, "latent_dim", 8)
        self.encoder_hidden_dims = getattr(configs, "encoder_hidden_dims", [32, 64])
        self.decoder_hidden_dims = getattr(configs, "decoder_hidden_dims", [64, 32])

        # 等变正则
        self.use_equivariance_regularization = getattr(configs, "use_equivariance_regularization", False)
        self.equivariance_weight = getattr(configs, "equivariance_weight", 0.1)
        self.equivariance_mode = getattr(configs, "equivariance_mode", "invariant")  # "invariant" or "permute"
        self.num_shifts_reg = getattr(configs, "num_shifts_reg", 3)

        # 预测器
        self.use_position_sensitive_predictor = getattr(configs, "use_position_sensitive_predictor", True)
        self.period_mlp_hidden = getattr(configs, "period_mlp_hidden", 128)
        self.period_mlp_dropout = getattr(configs, "period_mlp_dropout", 0.0)

        # 计算 period 数并做序列补齐信息
        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len

        # 需要填充到整 period 长度
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        # 初始化组件
        self._setup_encoders_decoders()
        self._setup_predictor()
        self.regularizer = EquivarianceRegularizer(self.equivariance_mode, self.num_shifts_reg)

    def _setup_encoders_decoders(self):
        self.encoder = CircularEquivariantEncoder(
            self.period_len, self.latent_dim, self.encoder_hidden_dims
        )
        self.decoder = CircularEquivariantDecoder(
            self.latent_dim, self.period_len, self.decoder_hidden_dims, L_enc=self.encoder.L_enc
        )

    def _setup_predictor(self):
        if not self.use_position_sensitive_predictor:
            raise ValueError("线性预测器已删除，请使用位置敏感预测器（本版为 ResMLP）")

        self.predictor = PeriodwiseResMLP(
            num_periods_input=self.num_periods_input,
            num_periods_output=self.num_periods_output,
            hidden=self.period_mlp_hidden,
            dropout=self.period_mlp_dropout
        )

    # -------- 编码/预测/解码 -----------
    def _encode_periods(self, x_periods):
        """x_periods: (B, C, P_in, L) -> (B, C, P_in, D)"""
        B, C, P, L = x_periods.shape
        x_flat = x_periods.reshape(B*C*P, L)
        z_flat = self.encoder(x_flat)
        return z_flat.view(B, C, P, self.latent_dim)

    def _predict_future_coeffs(self, encoded_coeffs):
        """(B, C, P_in, D) -> (B, C, P_out, D)"""
        return self.predictor(encoded_coeffs)

    def _decode_periods(self, coeffs):
        """(B, C, P, D) -> (B, C, P, L)"""
        B, C, P, D = coeffs.shape
        z_flat = coeffs.reshape(B*C*P, D)
        per_flat = self.decoder(z_flat)  # (B*C*P, L)
        return per_flat.view(B, C, P, self.period_len)

    def _compute_equivariance_loss(self, x_periods):
        if not self.use_equivariance_regularization:
            return torch.tensor(0.0, device=x_periods.device)
        return self.regularizer(self.encoder, x_periods, self.latent_dim, x_periods.device)

    # ------------- 前向 --------------
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        输入: x_enc (B, seq_len, C)
        输出: y_hat (B, pred_len, C), encoded_coeffs, future_coeffs
        """
        x = x_enc.float()                   # (B, L, C)
        B, L, C = x.shape
        x = x.permute(0, 2, 1)              # (B, C, L)

        # 统一采用环形补齐到整period
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode='circular')  # (B, C, total_len_in)

        # 切块为 period
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)  # (B,C,P_in,L)

        # 编码->预测->解码
        encoded_coeffs = self._encode_periods(x_periods)                   # (B,C,P_in,D)
        future_coeffs = self._predict_future_coeffs(encoded_coeffs)        # (B,C,P_out,D)
        decoded_periods = self._decode_periods(future_coeffs)              # (B,C,P_out,L)

        # 拼接并截断到 pred_len
        y_hat = decoded_periods.reshape(B, C, -1)[..., :self.pred_len]     # (B,C,pred_len)
        y_hat = y_hat.permute(0, 2, 1)                                     # (B,pred_len,C)
        return y_hat, encoded_coeffs, future_coeffs

    # --------- Lightning 步骤 ----------
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        dec_inp = self._build_decoder_input(batch_y)
        outputs, encoded_coeffs, _ = self(
            x_enc=batch_x, x_mark_enc=batch_x_mark, x_dec=dec_inp, x_mark_dec=batch_y_mark
        )

        # 统一用 self.pred_len
        outputs = outputs[:, -self.pred_len:, :]
        target = batch_y[:, -self.pred_len:, :]

        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        main_loss = criterion(outputs, target)

        # 等变/不变 正则
        if self.use_equivariance_regularization:
            # 构造与前向一致的 period 切块
            x = batch_x.permute(0, 2, 1)  # (B,C,L)
            if self.pad_seq_len > 0:
                x = F.pad(x, (0, self.pad_seq_len), mode='circular')
            x_periods = x.view(batch_x.size(0), self.enc_in, self.num_periods_input, self.period_len)
            eq_loss = self._compute_equivariance_loss(x_periods)
            loss = main_loss + self.equivariance_weight * eq_loss
            self.log("train_loss_equivariance", eq_loss, prog_bar=False, on_epoch=True)
        else:
            loss = main_loss

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
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
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
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