import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.pl_bases.default_module import DefaultPLModule
from src.models.layers.SelfAttention_Family import FullAttention, AttentionLayer



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
        num_heads: int = 4,
        dropout: float = 0.0,
        use_relpos: bool = True,
        period_len: int = 24,
        window_size: Optional[int] = None,
        attention_dim: Optional[int] = None,
        use_pos_embed: bool = False,
        pos_dropout: float = 0.0,
    ):
        super().__init__()
        # attention_dim参数对本实现不改变投影维度，但保持与CAPEN接口一致
        self.attention_dim = attention_dim or latent_dim
        assert self.attention_dim % num_heads == 0, "attention_dim 必须能被 num_heads 整除"

        self.latent_dim = latent_dim
        self.num_routers = num_routers
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.period_len = period_len

        # 可学习router
        self.router = nn.Parameter(torch.randn(num_routers, latent_dim))
        nn.init.trunc_normal_(self.router, std=0.02)

        # 相位位置嵌入（可选）
        if self.use_pos_embed:
            self.pos_embedding = nn.Parameter(torch.zeros(period_len, latent_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.pos_dropout = nn.Dropout(pos_dropout)

        # 两阶段注意力
        self.router_sender = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
            latent_dim,
            num_heads,
        )
        self.router_receiver = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
            latent_dim,
            num_heads,
        )

        # 层归一化与MLP
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, Z):  # Z: (B, C, L, D)
        B, C, L, D = Z.shape
        x = Z.view(B * C, L, D)

        # 位置嵌入（可选）
        if self.use_pos_embed:
            if L == self.period_len:
                pe = self.pos_embedding.unsqueeze(0).expand(B * C, -1, -1)
            elif L < self.period_len:
                pe = self.pos_embedding[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            else:
                repeat_factor = (L + self.period_len - 1) // self.period_len
                expanded_pe = self.pos_embedding.repeat(repeat_factor, 1)
                pe = expanded_pe[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            x = x + pe
            x = self.pos_dropout(x)

        # 阶段一：router聚合
        batch_router = self.router.unsqueeze(0).expand(B * C, -1, -1)  # (BC, R, D)
        router_buffer, _ = self.router_sender(batch_router, x, x, attn_mask=None)

        # 阶段二：router分发
        router_receive, _ = self.router_receiver(x, router_buffer, router_buffer, attn_mask=None)

        # 残差 + Norm
        out = x + self.dropout_layer(router_receive)
        out = self.norm1(out)

        # MLP 段 + 残差 + Norm
        mlp_out = self.mlp(out)
        out = out + self.dropout_layer(mlp_out)
        out = self.norm2(out)

        # 还原形状
        out = out.view(B, C, L, D)
        return out



class PhaseSeriesEncoder(nn.Module):
    def __init__(self, p_in: int, latent_dim: int, hidden: int = 32, use_mlp: bool = False, dropout: float = 0.0):
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
    def __init__(self, latent_dim: int, out_steps: int, hidden: int = 64, use_mlp: bool = False, dropout: float = 0.0):
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
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        # 归一化相关
        self.use_segment_norm = getattr(configs, "use_segment_norm", True)
        print("use_segment_norm", self.use_segment_norm)

        # 模块组装
        self.phase_encoder = PhaseSeriesEncoder(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim,
            hidden=self.phase_encoder_hidden,
            use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
            dropout=getattr(configs, "phase_encoder_dropout", 0.0),
        )

        # 仅使用降维注意力（DimReduction）
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

    # --------- 工具：相位重排 ----------
    @staticmethod
    def _to_phase_series(x_periods):
        """(B, C, P_in, L) -> (B, C, L, P_in)"""
        return x_periods.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _from_phase_steps_to_periods(y_phase_steps):
        """(B, C, L, P_out) -> (B, C, P_out, L)"""
        return y_phase_steps.permute(0, 1, 3, 2).contiguous()

    def _normalize_input(self, x, b, c):
        """
        输入归一化：支持segment_norm和全局归一化
        x: (B, C, L, P_in) when segment_norm is enabled
        """
        if self.use_segment_norm:
            # Segment归一化：对每个segment(沿P_in维)求均值
            segment_mean = torch.mean(x, dim=-1, keepdim=True)  # (B, C, L, 1)
            x = x - segment_mean
            return x, {"segment_mean": segment_mean}
        else:
            # 全局归一化：对每个变量的所有数据求均值
            x_flat = x.reshape(b, c, -1)
            mean = torch.mean(x_flat, dim=-1, keepdim=True)  # (b, c, 1)
            x_flat = x_flat - mean
            x = x_flat.reshape(b, c, self.period_len, self.num_periods_input)
            return x, {"mean": mean}

    def _denormalize_segment_phase(self, y_phase_steps, segment_mean):
        """
        段归一化的逆操作：将每个相位的均值加回到预测的相位步长上。
        y_phase_steps: (B, C, L, P_out), segment_mean: (B, C, L, 1)
        """
        return y_phase_steps + segment_mean

    # ------------- 前向 --------------
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        输入: x_enc (B, seq_len, C)
        输出: y_hat (B, pred_len, C), Z_encoded (B,C,L,D), future_phase_vals (B,C,L,P_out)
        """
        # 1) 基础处理
        x = x_enc.float()
        x = x.permute(0, 2, 1)  # (B, C, L_total)
        B, C, L_total = x.shape

        # 2) 环形补齐到整 period
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode='circular')  # (B, C, total_len_in)

        # 3) 切成 (B, C, P_in, L)
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)

        # 4) 相位并行：按相位取列 (B,C,L,P_in)
        phase_series = self._to_phase_series(x_periods)  # (B,C,L,P_in)

        # 4.5) Segment归一化（切片后）
        norm_stats = None
        if self.use_segment_norm:
            phase_series, norm_stats = self._normalize_input(phase_series, B, C)

        # 5) 相位编码 -> (B,C,L,D)
        Z = self.phase_encoder(phase_series)

        # 6) 相位轴交互：卷积/注意力/恒等
        Z = self.phase_interact(Z)  # (B,C,L,D)

        # 7) 逐相位预测 P_out 步
        y_phase_steps = self.predictor(Z)                             # (B,C,L,P_out)

        # 8) 段归一化的逆操作（如果需要）：对相位步长域执行
        if self.use_segment_norm and norm_stats is not None and "segment_mean" in norm_stats:
            y_phase_steps = self._denormalize_segment_phase(y_phase_steps, norm_stats["segment_mean"])  # (B,C,L,P_out)

        # 9) 还原为 (B,C,P_out,L) 并拼为序列
        y_periods = self._from_phase_steps_to_periods(y_phase_steps)  # (B,C,P_out,L)
        y_full = y_periods.reshape(B, C, -1)[..., :self.pred_len]     # (B,C,pred_len)
        y_hat = y_full.permute(0, 2, 1)                               # (B,pred_len,C)

        return y_hat, Z, y_phase_steps

    # --------- Lightning 步骤 ----------
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        # 工程兼容：仍然构造 dec_inp（虽未使用）
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


