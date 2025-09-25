import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.pl_bases.default_module import DefaultPLModule
from src.models.layers.SelfAttention_Family import FullAttention, AttentionLayer


# =========================================================
# RevIN 归一化模块
# =========================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization over time (per-sample, per-variable).
    对输入 x: (B, L, C) 在时间轴 L 做标准化；并提供反归一化。
    - 每个样本、每个变量独立计算均值/方差，适合时变分布。
    - affine=False 更稳健；若为 True，会在标准化后施加可学习仿射（不参与反归一化）。
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


# =========================================================
# 降维注意力模块 (Dimension Reduction Attention)
# =========================================================
class DimensionReductionAttention(nn.Module):
    """
    降维注意力模块：使用固定数量的router来降低attention复杂度。

    核心思想：
    1. 使用固定数量的router（远小于序列长度L）作为中介
    2. 第一阶段：router作为query，输入序列作为key/value，聚合信息到router
    3. 第二阶段：输入序列作为query，router作为key/value，分发信息回输入
    4. 总复杂度从O(L²)降低到O(L*R)，其中R是router数量

    输入/输出：Z: (B, C, L, D)
    """

    def __init__(
        self,
        latent_dim: int,
        num_routers: int = 8,  # router数量，通常远小于序列长度
        num_heads: int = 4,
        dropout: float = 0.0,
        period_len: int = 24,
        use_pos_embed: bool = False,
        pos_dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_routers = num_routers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.period_len = period_len

        # 可学习的router embeddings - 关键：这是固定大小的，不依赖于输入序列长度
        self.router = nn.Parameter(torch.randn(num_routers, latent_dim))
        nn.init.trunc_normal_(self.router, std=0.02)

        # 可学习的相位位置嵌入（在相位 L 维度上）
        if self.use_pos_embed:
            self.pos_embedding = nn.Parameter(torch.zeros(period_len, latent_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.pos_dropout = nn.Dropout(pos_dropout)

        # 第一阶段：router聚合信息 (router作为query，输入作为key/value)
        self.router_sender = AttentionLayer(
            FullAttention(
                False, factor=5, attention_dropout=dropout, output_attention=False
            ),
            latent_dim,
            num_heads,
        )

        # 第二阶段：router分发信息 (输入作为query，router作为key/value)
        self.router_receiver = AttentionLayer(
            FullAttention(
                False, factor=5, attention_dropout=dropout, output_attention=False
            ),
            latent_dim,
            num_heads,
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(latent_dim)
    
        # dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, Z):  # Z: (B, C, L, D)
        B, C, L, D = Z.shape

        # 将输入reshape为 (BC, L, D) 以便处理
        x = Z.view(B * C, L, D)  # (BC, L, D)

        # 添加相位位置嵌入（如启用）
        if self.use_pos_embed:
            if L == self.period_len:
                pe = self.pos_embedding.unsqueeze(0).expand(B * C, -1, -1)  # (BC, L, D)
            elif L < self.period_len:
                pe = self.pos_embedding[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            else:
                repeat_factor = (L + self.period_len - 1) // self.period_len
                expanded_pe = self.pos_embedding.repeat(
                    repeat_factor, 1
                )  # (repeat*L0, D)
                pe = expanded_pe[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            x = x + pe
            x = self.pos_dropout(x)

        # 扩展router到batch维度
        batch_router = self.router.unsqueeze(0).expand(B * C, -1, -1)  # (BC, R, D)

        # 第一阶段：router聚合信息
        # router作为query，输入序列作为key/value
        router_buffer, _ = self.router_sender(
            batch_router,  # queries: (BC, R, D)
            x,  # keys: (BC, L, D)
            x,  # values: (BC, L, D)
            attn_mask=None,
        )  # 输出: (BC, R, D)

        # 第二阶段：router分发信息
        # 输入序列作为query，router作为key/value
        router_receive, _ = self.router_receiver(
            x,  # queries: (BC, L, D)
            router_buffer,  # keys: (BC, R, D)
            router_buffer,  # values: (BC, R, D)
            attn_mask=None,
        )  # 输出: (BC, L, D)

        # 残差连接和层归一化
        out = x + self.dropout_layer(router_receive)
        out = self.norm1(out)

        # reshape回原始维度
        out = out.view(B, C, L, D)  # (B, C, L, D)
        return out


# =========================================================
# 相位序列编码器
# =========================================================
class PhaseSeriesEncoder(nn.Module):
    """
    输入: (B, C, L, P_in)
    输出: (B, C, L, latent_dim)
    """

    def __init__(
        self,
        p_in: int,
        latent_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p_in = p_in
        self.latent_dim = latent_dim
        self.norm = nn.LayerNorm(latent_dim)
        self.projection = nn.Linear(self.p_in, self.latent_dim, bias=False)

    def forward(self, phase_series):  # (B, C, L, P_in)
        z = self.norm(self.projection(phase_series))
        return z


# =========================================================
# 逐相位预测器
# =========================================================
class PhasePredictor(nn.Module):
    """
    对每个相位的 latent 做映射直接输出 P_out 步的未来值。
    输入: Z(B, C, L, D) -> 输出: y(B, C, L, P_out)

    可选：使用两层 MLP（latent_dim -> hidden -> out_steps）。
    """

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

    def forward(self, Z):  # Z: (B, C, L, latent_dim)
        y = self.projection(self.norm(Z))
        return y


# =========================================================
# CAPENRec 主体 (降维注意力版本)
# =========================================================
class CAPENRec(DefaultPLModule):
    """
    CAPENRec（Circular Autoencoder for Periodic Equivariant forecasting with Dimension Reduction）
    —— 使用降维注意力的相位并行预测模型

    核心特点：
    1. 使用固定数量的router降低attention复杂度
    2. 两阶段注意力：router聚合 → router分发
    3. 复杂度从O(L²)降低到O(L×R)，其中R是router数量
    """

    def __init__(self, configs):
        super().__init__(configs)

        # 基础参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len  # L

        # 维度与结构超参
        self.latent_dim = getattr(configs, "latent_dim", 8)
        self.phase_encoder_hidden = getattr(configs, "phase_encoder_hidden", 32)
        self.predictor_hidden = getattr(configs, "predictor_hidden", 64)

        # 降维注意力参数
        self.phase_num_routers = getattr(configs, "phase_num_routers", 8)  # router数量
        self.phase_attn_heads = getattr(configs, "phase_attn_heads", 1)
        self.phase_attn_dropout = getattr(configs, "phase_attn_dropout", 0.0)
        self.phase_use_pos_embed = getattr(configs, "phase_use_pos_embed", True)
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
            dropout=getattr(configs, "phase_encoder_dropout", 0.0),
        )

        # 降维注意力交互
        self.phase_interact = DimensionReductionAttention(
            latent_dim=self.latent_dim,
            num_routers=self.phase_num_routers,
            num_heads=self.phase_attn_heads,
            dropout=self.phase_attn_dropout,
            period_len=self.period_len,
            use_pos_embed=self.phase_use_pos_embed,
            pos_dropout=self.phase_pos_dropout,
        )

        self.predictor = PhasePredictor(
            latent_dim=self.latent_dim,
            out_steps=self.num_periods_output,
            hidden=self.predictor_hidden,
            use_mlp=getattr(configs, "predictor_use_mlp", True),
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

        # 5) 相位编码 -> (B,C,L,latent_dim)
        Z = self.phase_encoder(phase_series)

        # 6) 降维注意力交互
        Z = self.phase_interact(Z)  # (B,C,L,latent_dim)

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
