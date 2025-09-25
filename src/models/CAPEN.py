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
# 相位轴上的轻量环形深度可分卷积
# =========================================================
class DepthwiseCircularConv1dPhase(nn.Module):
    """
    在相位轴 L 上做 depthwise 1D 卷积，但仅按潜维 D 建组（groups = latent_dim）。
    做法：把变量 C 并到 batch 维 (B*C, D, L)，使同一套 D 个核在所有变量上共享。
    输入/输出：Z: (B, C, L, D)
    """
    def __init__(self, channels: int, latent_dim: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size 建议奇数"
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
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
        x = Z.permute(0, 1, 3, 2).reshape(B * C, D, L)  # (B*C, D, L)
        pad = self.kernel_size - 1
        x = F.pad(x, (pad, 0), mode="circular")         # 左侧环形补齐
        x = self.dw(x)                                  # (B*C, D, L)
        x = x.view(B, C, D, L).permute(0, 1, 3, 2)      # (B, C, L, D)
        return x


# =========================================================
# 相位注意力（新增）
# =========================================================
class CircularRelativePositionBias(nn.Module):
    """
    学习一个长度为 L 的相对位移偏置向量 b[d] (d in [0, L-1])，构成循环相对位置偏置矩阵：
    bias[i, j] = b[(i - j) mod L]
    可选: 窗口 window_size（环形局部注意力）；窗口外置为 -inf 屏蔽。
    """
    def __init__(self, num_heads: int, L: int, use_relpos: bool = True, window_size: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.L = L
        self.use_relpos = use_relpos
        self.window_size = window_size
        if use_relpos:
            self.rel_bias = nn.Parameter(torch.zeros(num_heads, L))  # b[h, d]
            nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, L: Optional[int] = None):
        H = self.num_heads
        L = L or self.L

        if not self.use_relpos:
            bias = torch.zeros(H, L, L)
        else:
            idx = torch.arange(L)
            rel = (idx[:, None] - idx[None, :]) % L  # (L, L)
            bias = self.rel_bias[:, rel]             # (H, L, L)

        if self.window_size is not None:
            w = self.window_size
            idx = torch.arange(L)
            dmat = (idx[:, None] - idx[None, :]).abs()  # 未取 mod
            d_circ = torch.minimum(dmat, L - dmat)      # (L, L)
            allow = (d_circ <= w)                       # True 保留
            mask = torch.where(
                allow,
                torch.zeros_like(d_circ, dtype=bias.dtype),
                torch.full_like(d_circ, float("-inf")),
            )
            bias = bias + mask.unsqueeze(0)  # (H, L, L)
        return bias  # (H, L, L)


class PhaseAttention(nn.Module):
    """
    在相位轴 L 上做多头自注意力（跨变量共享参数）。
    - 输入/输出：Z: (B, C, L, D)
    - 把 C 并到 batch： (B*C, L, D)
    - 支持：循环相对位置偏置 + 环形局部窗口（可选）
    - 支持：自定义attention计算维度（attention_dim），与latent_dim解耦
    """
    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_relpos: bool = True,
        period_len: int = 24,
        window_size: Optional[int] = None,
        attention_dim: Optional[int] = None,
    ):
        super().__init__()
        # 如果未指定attention_dim，则使用latent_dim
        self.attention_dim = attention_dim or latent_dim
        assert self.attention_dim % num_heads == 0, "attention_dim 必须能被 num_heads 整除"
        
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
        self.dropout = dropout

        # 线性投影：输入是latent_dim，QKV投影到attention_dim
        self.q_proj = nn.Linear(latent_dim, self.attention_dim, bias=False)
        self.k_proj = nn.Linear(latent_dim, self.attention_dim, bias=False)
        self.v_proj = nn.Linear(latent_dim, self.attention_dim, bias=False)
        # 输出投影：从attention_dim回到latent_dim
        self.o_proj = nn.Linear(self.attention_dim, latent_dim, bias=False)

        # 循环相对位置偏置
        self.relpos = CircularRelativePositionBias(
            num_heads=num_heads,
            L=period_len,
            use_relpos=use_relpos,
            window_size=window_size,
        )

    def _sdpa(self, q, k, v, attn_mask=None, dropout_p=0.0, training=False):
        """
        形状: q,k,v: (N, L, Dh); attn_mask(可选): (N, L, L) 加性偏置（含 -inf 位置）。
        返回: (N, L, Dh)
        """
        # 优先使用官方 API（若存在）
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p if training else 0.0,
                is_causal=False,
            )

        # 手写回落实现（等价于上面）
        Dh = q.shape[-1]
        # logits: (N, L, L)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
        if attn_mask is not None:
            # 要求同 dtype/device；这里强制对齐
            attn_mask = attn_mask.to(dtype=logits.dtype, device=logits.device)
            logits = logits + attn_mask
        attn = torch.softmax(logits, dim=-1)
        if dropout_p and training:
            attn = F.dropout(attn, p=dropout_p, training=True)
        out = torch.matmul(attn, v)  # (N, L, Dh)
        return out    

    def forward(self, Z):  # Z: (B, C, L, D)
        B, C, L, D = Z.shape
        x = Z.view(B * C, L, D)  # (BC, L, D)

        # Q, K, V: 投影到attention_dim
        q = self.q_proj(x)  # (BC, L, attention_dim)
        k = self.k_proj(x)  # (BC, L, attention_dim)
        v = self.v_proj(x)  # (BC, L, attention_dim)

        H, Dh = self.num_heads, self.head_dim
        q = q.view(B * C, L, H, Dh).permute(0, 2, 1, 3)  # (BC, H, L, Dh)
        k = k.view(B * C, L, H, Dh).permute(0, 2, 1, 3)
        v = v.view(B * C, L, H, Dh).permute(0, 2, 1, 3)

        # 相对位置偏置（含 -inf 窗口屏蔽时）
        rel_bias = self.relpos(L=L).to(Z.device)                        # (H, L, L)
        rel_bias = rel_bias.unsqueeze(0).expand(B * C, -1, -1, -1)      # (BC, H, L, L)
        rel_bias = rel_bias.reshape(B * C * H, L, L)                    # (BC*H, L, L)

        # 合并 batch 和 head 维，调用 sdpa
        q_ = q.reshape(B * C * H, L, Dh)
        k_ = k.reshape(B * C * H, L, Dh)
        v_ = v.reshape(B * C * H, L, Dh)

        out = self._sdpa(
            q_, k_, v_,
            attn_mask=rel_bias,  # 作为 logits 加性偏置，-inf 位置被屏蔽
            dropout_p=self.dropout,
            training=self.training,
        )  # (BC*H, L, Dh)

        out = out.view(B * C, H, L, Dh).permute(0, 2, 1, 3).contiguous()  # (BC, L, H, Dh)
        out = out.view(B * C, L, self.attention_dim)                      # (BC, L, attention_dim)
        out = self.o_proj(out)                                            # (BC, L, latent_dim)
        out = out.view(B, C, L, D)                                        # (B, C, L, D)
        return out


class DimensionReductionAttention(nn.Module):
    """
    降维注意力模块：参考Crossformer的思想，使用固定数量的router来降低attention复杂度。
    
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
        use_relpos: bool = True,
        period_len: int = 24,
        window_size: Optional[int] = None,
        attention_dim: Optional[int] = None,
        use_pos_embed: bool = False,
        pos_dropout: float = 0.0,
    ):
        super().__init__()
        # 如果未指定attention_dim，则使用latent_dim
        self.attention_dim = attention_dim or latent_dim
        assert self.attention_dim % num_heads == 0, "attention_dim 必须能被 num_heads 整除"
        
        self.latent_dim = latent_dim
        self.num_routers = num_routers
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
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
                False, 
                factor=5, 
                attention_dropout=dropout, 
                output_attention=False
            ),
            latent_dim,
            num_heads
        )

        # 第二阶段：router分发信息 (输入作为query，router作为key/value)  
        self.router_receiver = AttentionLayer(
            FullAttention(
                False,
                factor=5,
                attention_dropout=dropout,
                output_attention=False
            ),
            latent_dim, 
            num_heads
        )

        # 层归一化和MLP
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim)
        )
        
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
                expanded_pe = self.pos_embedding.repeat(repeat_factor, 1)  # (repeat*L0, D)
                pe = expanded_pe[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            x = x + pe
            x = self.pos_dropout(x)
        
        # 扩展router到batch维度
        batch_router = self.router.unsqueeze(0).expand(B * C, -1, -1)  # (BC, R, D)
        
        # 第一阶段：router聚合信息
        # router作为query，输入序列作为key/value
        router_buffer, _ = self.router_sender(
            batch_router,  # queries: (BC, R, D) 
            x,            # keys: (BC, L, D)
            x,            # values: (BC, L, D)
            attn_mask=None
        )  # 输出: (BC, R, D)

        # 第二阶段：router分发信息  
        # 输入序列作为query，router作为key/value
        router_receive, _ = self.router_receiver(
            x,             # queries: (BC, L, D)
            router_buffer, # keys: (BC, R, D) 
            router_buffer, # values: (BC, R, D)
            attn_mask=None
        )  # 输出: (BC, L, D)
        
        # 残差连接和层归一化
        out = x + self.dropout_layer(router_receive)
        out = self.norm1(out)
        
        # MLP层
        mlp_out = self.mlp(out)
        out = out + self.dropout_layer(mlp_out)
        out = self.norm2(out)
        
        # reshape回原始维度
        out = out.view(B, C, L, D)  # (B, C, L, D)
        return out


class FixedAttention(nn.Module):
    """
    固定注意力模块：QK矩阵设置为固定的带衰减权重的对角阵，而非计算得出。
    
    核心思想：
    1. 不计算QK^T，而是使用预定义的注意力权重矩阵
    2. 注意力权重以对角线为中心，随距离衰减
    3. 支持环形衰减（考虑周期性）
    4. 只需要计算V投影，大大减少计算量
    
    输入/输出：Z: (B, C, L, D)
    """
    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        period_len: int = 24,
        decay_rate: float = 0.1,  # 衰减率，距离每增加1，权重乘以exp(-decay_rate)
        max_distance: Optional[int] = None,  # 最大有效距离，超过此距离权重为0
        attention_dim: Optional[int] = None,
    ):
        super().__init__()
        # 如果未指定attention_dim，则使用latent_dim
        self.attention_dim = attention_dim or latent_dim
        assert self.attention_dim % num_heads == 0, "attention_dim 必须能被 num_heads 整除"
        
        self.latent_dim = latent_dim
        self.attention_dim = attention_dim or latent_dim
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
        self.dropout = dropout
        self.period_len = period_len
        self.decay_rate = decay_rate
        self.max_distance = max_distance or (period_len // 2)

        # 只需要V投影，不需要Q和K投影
        self.v_proj = nn.Linear(latent_dim, self.attention_dim, bias=False)
        # 输出投影
        self.o_proj = nn.Linear(self.attention_dim, latent_dim, bias=False)

        # 预计算固定的注意力权重矩阵
        self.register_buffer('fixed_attn_weights', self._create_fixed_attention_weights())

    def _create_fixed_attention_weights(self):
        """
        创建固定的注意力权重矩阵 (H, L, L)
        权重以对角线为中心，随环形距离衰减
        """
        L = self.period_len
        H = self.num_heads
        
        # 创建位置索引
        i_idx = torch.arange(L).unsqueeze(1)  # (L, 1)
        j_idx = torch.arange(L).unsqueeze(0)  # (1, L)
        
        # 计算环形距离
        diff = (i_idx - j_idx).abs()  # (L, L)
        circular_dist = torch.minimum(diff, L - diff)  # 环形距离
        
        # 计算衰减权重
        weights = torch.exp(-self.decay_rate * circular_dist.float())  # (L, L)
        
        # 应用最大距离限制
        if self.max_distance is not None:
            mask = circular_dist <= self.max_distance
            weights = weights * mask.float()
        
        # 为每个head创建不同的权重（可选：这里使用相同权重）
        # 可以为不同head设置不同的衰减率或偏移
        weights = weights.unsqueeze(0).expand(H, -1, -1)  # (H, L, L)
        
        # 可选：为不同head添加轻微扰动
        if H > 1:
            head_scales = torch.linspace(0.8, 1.2, H).unsqueeze(1).unsqueeze(2)
            weights = weights * head_scales
        
        # 归一化（沿最后一维softmax）
        weights = F.softmax(weights, dim=-1)
        
        return weights

    def forward(self, Z):  # Z: (B, C, L, D)
        B, C, L, D = Z.shape
        x = Z.view(B * C, L, D)  # (BC, L, D)

        # 只需要计算V投影
        v = self.v_proj(x)  # (BC, L, attention_dim)

        H, Dh = self.num_heads, self.head_dim
        v = v.view(B * C, L, H, Dh).permute(0, 2, 1, 3)  # (BC, H, L, Dh)

        # 获取固定的注意力权重
        # 如果输入序列长度与预设不同，需要调整权重矩阵
        if L != self.period_len:
            # 简单处理：截取或循环扩展
            if L < self.period_len:
                attn_weights = self.fixed_attn_weights[:, :L, :L]
            else:
                # 循环扩展权重矩阵
                repeat_factor = (L + self.period_len - 1) // self.period_len
                expanded_weights = self.fixed_attn_weights.repeat(1, repeat_factor, repeat_factor)
                attn_weights = expanded_weights[:, :L, :L]
                # 重新归一化
                attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            attn_weights = self.fixed_attn_weights  # (H, L, L)

        # 扩展到batch维度
        attn_weights = attn_weights.unsqueeze(0).expand(B * C, -1, -1, -1)  # (BC, H, L, L)

        # 应用dropout（如果需要）
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        # 应用注意力权重
        # attn_weights: (BC, H, L, L), v: (BC, H, L, Dh)
        out = torch.matmul(attn_weights, v)  # (BC, H, L, Dh)

        # 输出处理
        out = out.permute(0, 2, 1, 3).contiguous()  # (BC, L, H, Dh)
        out = out.view(B * C, L, self.attention_dim)  # (BC, L, attention_dim)
        out = self.o_proj(out)  # (BC, L, latent_dim)
        out = out.view(B, C, L, D)  # (B, C, L, D)
        return out


class LearnableFixedAttention(nn.Module):
    """
    可学习的固定注意力模块：基于固定模式初始化，但允许通过训练更新注意力权重。
    
    核心思想：
    1. 使用固定的衰减模式作为初始化
    2. 将注意力权重设为可训练参数
    3. 提供开关控制是否允许训练更新
    4. 保持时间序列的局部性和周期性归纳偏置
    
    输入/输出：Z: (B, C, L, D)
    """
    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        period_len: int = 24,
        decay_rate: float = 0.1,  # 初始化的衰减率
        max_distance: Optional[int] = None,  # 最大有效距离
        attention_dim: Optional[int] = None,
        learnable: bool = True,  # 是否允许训练更新权重
        init_noise: float = 0.1,  # 初始化时添加的噪声强度
    ):
        super().__init__()
        # 如果未指定attention_dim，则使用latent_dim
        self.attention_dim = attention_dim or latent_dim
        assert self.attention_dim % num_heads == 0, "attention_dim 必须能被 num_heads 整除"
        
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
        self.dropout = dropout
        self.period_len = period_len
        self.decay_rate = decay_rate
        self.max_distance = max_distance or (period_len // 2)
        self.learnable = learnable
        self.init_noise = init_noise

        # 只需要V投影，不需要Q和K投影
        self.v_proj = nn.Linear(latent_dim, self.attention_dim, bias=False)
        # 输出投影
        self.o_proj = nn.Linear(self.attention_dim, latent_dim, bias=False)

        # 创建注意力权重 - 关键区别：可训练 vs 固定
        if self.learnable:
            # 可训练的注意力权重参数
            init_weights = self._create_initial_attention_weights()
            self.attn_weights = nn.Parameter(init_weights)
        else:
            # 固定的注意力权重buffer
            self.register_buffer('attn_weights', self._create_initial_attention_weights())

    def _create_initial_attention_weights(self):
        """
        创建初始的注意力权重矩阵 (H, L, L)
        使用衰减模式初始化，可选择性地添加噪声
        """
        L = self.period_len
        H = self.num_heads
        
        # 创建位置索引
        i_idx = torch.arange(L).unsqueeze(1)  # (L, 1)
        j_idx = torch.arange(L).unsqueeze(0)  # (1, L)
        
        # 计算环形距离
        diff = (i_idx - j_idx).abs()  # (L, L)
        circular_dist = torch.minimum(diff, L - diff)  # 环形距离
        
        # 计算衰减权重
        weights = torch.exp(-self.decay_rate * circular_dist.float())  # (L, L)
        
        # 应用最大距离限制
        if self.max_distance is not None:
            mask = circular_dist <= self.max_distance
            weights = weights * mask.float()
        
        # 为每个head创建不同的权重
        weights = weights.unsqueeze(0).expand(H, -1, -1)  # (H, L, L)
        
        # 为不同head添加不同的特性
        if H > 1:
            # 不同head有不同的衰减率
            head_decay_rates = torch.linspace(0.5, 2.0, H).unsqueeze(1).unsqueeze(2)
            head_weights = torch.exp(-head_decay_rates * circular_dist.float().unsqueeze(0))
            
            # 应用距离限制
            if self.max_distance is not None:
                mask = circular_dist.unsqueeze(0) <= self.max_distance
                head_weights = head_weights * mask.float()
            
            weights = head_weights
        
        # 添加初始化噪声（如果可学习）
        if self.learnable and self.init_noise > 0:
            noise = torch.randn_like(weights) * self.init_noise
            weights = weights + noise
            # 确保权重非负
            weights = F.relu(weights)
        
        # 归一化（沿最后一维softmax）
        # 注意：如果是可学习的，我们在forward中再做softmax以保持可微性
        if not self.learnable:
            weights = F.softmax(weights, dim=-1)
        
        return weights

    def forward(self, Z):  # Z: (B, C, L, D)
        B, C, L, D = Z.shape
        x = Z.view(B * C, L, D)  # (BC, L, D)

        # 只需要计算V投影
        v = self.v_proj(x)  # (BC, L, attention_dim)

        H, Dh = self.num_heads, self.head_dim
        v = v.view(B * C, L, H, Dh).permute(0, 2, 1, 3)  # (BC, H, L, Dh)

        # 获取注意力权重
        if L != self.period_len:
            # 处理不同序列长度的情况
            if L < self.period_len:
                raw_weights = self.attn_weights[:, :L, :L]
            else:
                # 循环扩展权重矩阵
                repeat_factor = (L + self.period_len - 1) // self.period_len
                expanded_weights = self.attn_weights.repeat(1, repeat_factor, repeat_factor)
                raw_weights = expanded_weights[:, :L, :L]
        else:
            raw_weights = self.attn_weights  # (H, L, L)

        # 如果是可学习的，需要在这里做softmax以保持可微性
        if self.learnable:
            attn_weights = F.softmax(raw_weights, dim=-1)
        else:
            attn_weights = raw_weights

        # 扩展到batch维度
        attn_weights = attn_weights.unsqueeze(0).expand(B * C, -1, -1, -1)  # (BC, H, L, L)

        # 应用dropout（如果需要）
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        # 应用注意力权重
        # attn_weights: (BC, H, L, L), v: (BC, H, L, Dh)
        out = torch.matmul(attn_weights, v)  # (BC, H, L, Dh)

        # 输出处理
        out = out.permute(0, 2, 1, 3).contiguous()  # (BC, L, H, Dh)
        out = out.view(B * C, L, self.attention_dim)  # (BC, L, attention_dim)
        out = self.o_proj(out)  # (BC, L, latent_dim)
        out = out.view(B, C, L, D)  # (B, C, L, D)
        return out

    def get_attention_weights(self):
        """
        获取当前的注意力权重（用于可视化和分析）
        """
        if self.learnable:
            return F.softmax(self.attn_weights, dim=-1)
        else:
            return self.attn_weights

    def freeze_weights(self):
        """冻结注意力权重，使其不可训练"""
        if hasattr(self, 'attn_weights') and isinstance(self.attn_weights, nn.Parameter):
            self.attn_weights.requires_grad = False

    def unfreeze_weights(self):
        """解冻注意力权重，使其可训练"""
        if hasattr(self, 'attn_weights') and isinstance(self.attn_weights, nn.Parameter):
            self.attn_weights.requires_grad = True


# =========================================================
# 相位序列编码器
# =========================================================
class PhaseSeriesEncoder(nn.Module):
    """
    对“每个相位的跨周期序列（长度 P_in）”做共享编码 → latent_dim
    输入: (B, C, L, P_in) -> 输出: (B, C, L, D)
    
    可选：使用两层 MLP（p_in -> hidden -> latent_dim）提升表征能力。
    """
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


# =========================================================
# 逐相位预测器
# =========================================================
class PhasePredictor(nn.Module):
    """
    对每个相位的 latent 做映射直接输出 P_out 步的未来值。
    输入: Z(B, C, L, D) -> 输出: y(B, C, L, P_out)
    
    可选：使用两层 MLP（latent_dim -> hidden -> out_steps）。
    """
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
        y = self.projection(Zn)  # (B, C, L, P_out)
        return y


# =========================================================
# CAPEN 主体
# =========================================================
class CAPEN(DefaultPLModule):
    """
    CAPEN（Circular Autoencoder for Periodic Equivariant forecasting, phase-parallel）
    —— 相位并行：每个相位跨周期编码 -> 预测该相位未来 P_out 步 -> 拼成未来序列
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

        _enc_hid_list = getattr(configs, "encoder_hidden_dims", None)
        self.phase_encoder_hidden = getattr(
            configs, "phase_encoder_hidden",
            (_enc_hid_list[0] if isinstance(_enc_hid_list, (list, tuple)) and len(_enc_hid_list) > 0 else 32)
        )

        # —— 相位交互模式与参数（新增）——
        self.phase_interaction_mode = getattr(configs, "phase_interaction_mode", "conv")  # "conv"|"attn"|"dim_reduction"|"fixed"|"learnable_fixed"|"none"
        self.use_phase_interaction = getattr(configs, "use_phase_interaction", True)

        # 卷积参数（沿用）
        self.phase_kernel_size = getattr(configs, "phase_kernel_size", 3)

        # 注意力参数（新增和扩展）
        self.phase_attn_heads = getattr(configs, "phase_attn_heads", 4)
        self.phase_attn_dropout = getattr(configs, "phase_attn_dropout", 0.0)
        self.phase_attn_use_relpos = getattr(configs, "phase_attn_use_relpos", True)
        self.phase_attn_window = getattr(configs, "phase_attn_window", None)  # e.g., 2 表示 ±2 环形局部注意
        self.phase_attention_dim = getattr(configs, "phase_attention_dim", None)  # 内在attention计算维度
        
        # 降维注意力参数
        self.phase_num_routers = getattr(configs, "phase_num_routers", 8)  # router数量
        
        # 固定注意力参数
        self.phase_decay_rate = getattr(configs, "phase_decay_rate", 0.1)  # 衰减率
        self.phase_max_distance = getattr(configs, "phase_max_distance", None)  # 最大有效距离
        
        # 可学习固定注意力参数
        self.phase_learnable_fixed = getattr(configs, "phase_learnable_fixed", True)  # 是否允许训练更新
        self.phase_init_noise = getattr(configs, "phase_init_noise", 0.1)  # 初始化噪声强度

        # 预测器隐藏维（兼容 predictor_hidden）
        self.predictor_hidden = getattr(configs, "predictor_hidden", 64)

        # 等变正则（占位）
        self.use_equivariance_regularization = getattr(configs, "use_equivariance_regularization", False)
        self.equivariance_weight = getattr(configs, "equivariance_weight", 0.1)
        self.num_shifts_reg = getattr(configs, "num_shifts_reg", 2)

        # 计算周期数（输入/输出）
        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        # 归一化相关
        self.use_revin = getattr(configs, "use_revin", True)
        self.revin_affine = getattr(configs, "revin_affine", False)
        self.revin_eps = getattr(configs, "revin_eps", 1e-5)
        if self.use_revin:
            self.revin = RevIN(num_features=self.enc_in, eps=self.revin_eps, affine=self.revin_affine)

        # 模块组装
        self.phase_encoder = PhaseSeriesEncoder(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim,
            hidden=self.phase_encoder_hidden,
            use_mlp=getattr(configs, "phase_encoder_use_mlp", True),
            dropout=getattr(configs, "phase_encoder_dropout", 0.0),
        )

        # 相位交互（根据模式选择）
        if self.use_phase_interaction and self.phase_interaction_mode == "conv":
            self.phase_interact = DepthwiseCircularConv1dPhase(
                channels=self.enc_in, latent_dim=self.latent_dim, kernel_size=self.phase_kernel_size
            )
        elif self.use_phase_interaction and self.phase_interaction_mode == "attn":
            self.phase_interact = PhaseAttention(
                latent_dim=self.latent_dim,
                num_heads=self.phase_attn_heads,
                dropout=self.phase_attn_dropout,
                use_relpos=self.phase_attn_use_relpos,
                period_len=self.period_len,
                window_size=self.phase_attn_window,
                attention_dim=self.phase_attention_dim,
            )
        elif self.use_phase_interaction and self.phase_interaction_mode == "dim_reduction":
            self.phase_interact = DimensionReductionAttention(
                latent_dim=self.latent_dim,
                num_routers=self.phase_num_routers,
                num_heads=self.phase_attn_heads,
                dropout=self.phase_attn_dropout,
                use_relpos=self.phase_attn_use_relpos,
                period_len=self.period_len,
                window_size=self.phase_attn_window,
                attention_dim=self.phase_attention_dim,
                use_pos_embed=getattr(configs, "phase_use_pos_embed", False),
                pos_dropout=getattr(configs, "phase_pos_dropout", 0.0),
            )
        elif self.use_phase_interaction and self.phase_interaction_mode == "fixed":
            self.phase_interact = FixedAttention(
                latent_dim=self.latent_dim,
                num_heads=self.phase_attn_heads,
                dropout=self.phase_attn_dropout,
                period_len=self.period_len,
                decay_rate=self.phase_decay_rate,
                max_distance=self.phase_max_distance,
                attention_dim=self.phase_attention_dim,
            )
        elif self.use_phase_interaction and self.phase_interaction_mode == "learnable_fixed":
            self.phase_interact = LearnableFixedAttention(
                latent_dim=self.latent_dim,
                num_heads=self.phase_attn_heads,
                dropout=self.phase_attn_dropout,
                period_len=self.period_len,
                decay_rate=self.phase_decay_rate,
                max_distance=self.phase_max_distance,
                attention_dim=self.phase_attention_dim,
                learnable=self.phase_learnable_fixed,
                init_noise=self.phase_init_noise,
            )
        else:
            self.phase_interact = nn.Identity()

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
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
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
            x = F.pad(x, (0, self.pad_seq_len), mode='circular')  # (B, C, total_len_in)

        # 3) 切成 (B, C, P_in, L)
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)

        # 4) 相位并行：按相位取列 (B,C,L,P_in)
        phase_series = self._to_phase_series(x_periods)  # (B,C,L,P_in)

        # 5) 相位编码 -> (B,C,L,D)
        Z = self.phase_encoder(phase_series)

        # 6) 相位轴交互：卷积/注意力/恒等
        Z = self.phase_interact(Z)  # (B,C,L,D)

        # 7) 逐相位预测 P_out 步
        y_phase_steps = self.predictor(Z)                             # (B,C,L,P_out)

        # 8) 还原为 (B,C,P_out,L) 并拼为序列
        y_periods = self._from_phase_steps_to_periods(y_phase_steps)  # (B,C,P_out,L)
        y_full = y_periods.reshape(B, C, -1)[..., :self.pred_len]     # (B,C,pred_len)
        y_hat = y_full.permute(0, 2, 1)                               # (B,pred_len,C)

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