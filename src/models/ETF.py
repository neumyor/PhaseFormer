import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Any, Tuple


class PositionalEncoding(nn.Module):
    """位置编码，为输入添加位置信息"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class Normalization(nn.Module):
    """归一化层，可以根据配置选择不同的归一化方法"""

    def __init__(self, d_model: int, norm_type: str):
        super().__init__()
        if norm_type == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(d_model)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm1d(d_model)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.norm, nn.BatchNorm1d) or isinstance(
            self.norm, nn.InstanceNorm1d
        ):
            # 调整维度以适应批归一化和实例归一化的输入要求
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            return self.norm(x)


class CrossAttentionLayer(nn.Module):
    """交叉注意力层，用于融合参考变量和目标变量的特征"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_type: str,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层和解耦
        self.norm1 = Normalization(d_model, norm_type)
        self.norm2 = Normalization(d_model, norm_type)
        self.norm3 = Normalization(d_model, norm_type)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 激活函数
        self.activation = F.relu if activation == "relu" else F.gelu

        # 存储注意力分数
        self.attention_scores = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: 目标序列，形状 [tgt_seq_len, batch_size, d_model]
            memory: 参考序列，形状 [memory_seq_len, batch_size, d_model]
        """
        # 目标序列的自注意力
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 交叉注意力：目标序列关注参考序列
        tgt2, attn_weights = self.cross_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        self.attention_scores = attn_weights  # 存储注意力分数

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SelfAttentionLayer(nn.Module):
    """自注意力层，用于细化目标变量的特征"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_type: str,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = Normalization(d_model, norm_type)
        self.norm2 = Normalization(d_model, norm_type)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: 输入序列，形状 [seq_len, batch_size, d_model]
        """
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class ExternalVariableEncoder(nn.Module):
    """外部变量编码器，将多个参考变量编码为token"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len

        # 共享的编码器
        self.encoder = nn.Sequential(
            nn.Linear(configs.seq_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, external_vars: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            external_vars: 外部变量列表，每个元素形状为 [batch_size, seq_len, n_external_vars]
        Returns:
            encoded_external: 编码后的外部变量，形状为 [batch_size, n_external_vars, d_model]
        """
        
        external_vars = external_vars.transpose(1, 2)  # [batch_size, n_external_vars, seq_len]

        # 对每个外部变量进行编码
        encoded_vars = [self.encoder(var) for var in external_vars]

        # 合并所有外部变量的编码
        encoded_external = torch.stack(
            encoded_vars, dim=0
        )  # [batch_size, n_external_vars, d_model]

        return encoded_external


class TargetSequenceEncoder(nn.Module):
    """基于PatchTST分块思想的目标序列编码层"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.target_dim = configs.target_dim
        self.patch_size = configs.patch_size

        # 确保序列长度可被分块大小整除
        assert self.seq_len % self.patch_size == 0, "序列长度必须是分块大小的整数倍"
        self.n_patches = self.seq_len // self.patch_size  # token数量由分块数决定

        # Patch嵌入层：[patch_size*target_dim] -> d_model
        self.patch_embedding = nn.Linear(
            self.patch_size * self.target_dim, self.d_model
        )

        # 位置编码（针对分块位置）
        self.pos_encoder = PositionalEncoding(self.d_model)

    def forward(self, target_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_seq: 目标序列，形状 [batch_size, seq_len, target_dim]
        Returns:
            encoded_target: 编码后的目标token序列，形状 [batch_size, n_patches, d_model]
        """
        batch_size = target_seq.size(0)

        # 1. 序列分块：[batch, seq_len, target_dim] -> [batch, n_patches, patch_size, target_dim]
        patches = target_seq.view(
            batch_size, self.n_patches, self.patch_size, self.target_dim
        )

        # 2. 展平每个patch并嵌入：[batch, n_patches, patch_size*target_dim] -> [batch, n_patches, d_model]
        patch_emb = self.patch_embedding(patches.flatten(2))  # 展平每个patch的时空维度

        # 3. 添加分块位置编码
        encoded_patches = self.pos_encoder(patch_emb)  # [batch, n_patches, d_model]

        return encoded_patches


class FeatureFusionLayer(nn.Module):
    """特征融合层，将参考变量对应的token与目标变量对应的序列token进行Cross-Attention"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.nhead = configs.nhead
        self.dim_feedforward = configs.dim_feedforward
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.norm_type = configs.norm_type

        # 交叉注意力层
        self.cross_attention = CrossAttentionLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            self.activation,
            self.norm_type,
        )

    def forward(
        self, target_tokens: torch.Tensor, external_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            target_tokens: 目标序列token，形状 [batch_size, n_target_tokens, d_model]
            external_tokens: 外部变量token，形状 [batch_size, n_external_vars, d_model]
        Returns:
            fused_features: 融合后的特征，形状 [batch_size, n_target_tokens, d_model]
        """
        # 调整维度以适应PyTorch的注意力模块要求 [seq_len, batch_size, d_model]
        target_tokens_t = target_tokens.transpose(0, 1)
        external_tokens_t = external_tokens.transpose(0, 1)

        # 应用交叉注意力
        fused = self.cross_attention(
            target_tokens_t,
            external_tokens_t,
            memory_mask=None,
            memory_key_padding_mask=None,
        )

        # 转回原始维度顺序
        return fused.transpose(0, 1)  # [batch_size, n_target_tokens, d_model]

    def get_attention_scores(self):
        """获取注意力分数"""
        return self.cross_attention.attention_scores


class FeatureRefinementLayer(nn.Module):
    """特征细化层，将目标变量对应的序列token进行SelfAttention"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.nhead = configs.nhead
        self.dim_feedforward = configs.dim_feedforward
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.norm_type = configs.norm_type

        # 自注意力层
        self.self_attention = SelfAttentionLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            self.activation,
            self.norm_type,
        )

    def forward(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_tokens: 目标序列token，形状 [batch_size, n_target_tokens, d_model]
        Returns:
            refined_features: 细化后的特征，形状 [batch_size, n_target_tokens, d_model]
        """
        # 调整维度以适应PyTorch的注意力模块要求 [seq_len, batch_size, d_model]
        target_tokens_t = target_tokens.transpose(0, 1)

        # 应用自注意力
        refined = self.self_attention(target_tokens_t)

        # 转回原始维度顺序
        return refined.transpose(0, 1)  # [batch_size, n_target_tokens, d_model]


class ETF(nn.Module):
    """
    ETF (Externally Tuned Fusion) 多元时间序列预测模型

    该模型通过交叉注意力机制融合多个外部变量和目标时间序列，
    并通过交替的特征融合层和特征细化层进行特征提取和预测。
    """

    def __init__(self, configs, *args, **kwargs):
        super().__init__()
        self.configs = configs

        # 目标变量索引
        self.target_var_index = configs.target_var_index

        # 控制是否保存注意力分数的开关
        self.save_attention_scores = getattr(configs, "save_attention_scores", False)

        # 模型参数
        self.d_model = configs.d_model
        self.n_layers = configs.n_layers  # 特征融合层和特征细化层的对数
        self.pred_len = configs.pred_len
        self.target_dim = configs.target_dim

        # 编码器
        self.target_encoder = TargetSequenceEncoder(configs)
        self.external_encoder = ExternalVariableEncoder(configs)

        # 特征融合和细化层
        self.fusion_layers = nn.ModuleList(
            [FeatureFusionLayer(configs) for _ in range(self.n_layers)]
        )
        self.refinement_layers = nn.ModuleList(
            [FeatureRefinementLayer(configs) for _ in range(self.n_layers)]
        )

        # 输出层
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.pred_len * self.target_dim),
        )

        # 存储所有层的注意力分数
        self.all_attention_scores = None

    def forward(
        self,
        x_enc,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            target_seq: 目标时间序列，形状 [batch_size, seq_len, target_dim]
            external_vars: 外部变量列表，每个元素形状为 [batch_size, seq_len, var_dim]
        Returns:
            prediction: 预测结果，形状 [batch_size, pred_len, target_dim]
        """
        # 分割目标序列和外部变量
        target_seq = x_enc[:, :, self.target_var_index]
        external_vars = torch.cat(
            [
                x_enc[:, :, : self.target_var_index],
                x_enc[:, :, self.target_var_index :],
            ],
            dim=2,
        )

        # 编码目标序列和外部变量
        target_tokens = self.target_encoder(
            target_seq
        )  # [batch_size, n_tokens, d_model]
        external_tokens = self.external_encoder(
            external_vars
        )  # [batch_size, n_external_vars, d_model]

        # 仅当开关开启时初始化注意力分数列表
        self.all_attention_scores = [] if self.save_attention_scores else None

        # 交替应用特征融合层和特征细化层
        for i in range(self.n_layers):
            # 特征融合
            fused_features = self.fusion_layers[i](target_tokens, external_tokens)

            # 仅当开关开启时记录注意力分数
            if self.save_attention_scores:
                attn_scores = self.fusion_layers[i].get_attention_scores()
                if attn_scores is not None:
                    # 对多头注意力分数求平均 [batch, tgt_len, src_len]
                    avg_attn_scores = attn_scores.mean(dim=1)  # 平均多头注意力
                    if self.all_attention_scores is not None:
                        self.all_attention_scores.append(avg_attn_scores.detach())

            # 特征细化
            target_tokens = self.refinement_layers[i](fused_features)

        # 取最后一个token进行预测
        final_representation = target_tokens[:, -1, :]  # [batch_size, d_model]

        # 投影到预测空间
        prediction = self.projection(
            final_representation
        )  # [batch_size, pred_len * target_dim]

        # 重塑为最终输出形状
        return prediction.reshape(-1, self.pred_len, self.target_dim)

    def get_external_attention_scores(self):
        """获取所有外部变量融合层的注意力分数矩阵"""
        return self.all_attention_scores

    def visualize_external_impact(self, save_path=None):
        """
        可视化在不同的外部变量融合层中，不同外部变量对应的token对于最终目标序列的影响程度的变化

        Args:
            external_var_names: 外部变量名称列表
            save_path: 图片保存路径，如果为None则显示图片
        """
        if not self.save_attention_scores or not self.all_attention_scores:
            raise ValueError(
                "请先开启save_attention_scores参数并运行模型以收集注意力分数"
            )

        # 假设我们关注最后一个目标token对外部变量的注意力
        # 注意力分数形状: [batch_size, tgt_len, src_len]
        batch_size = self.all_attention_scores[0].size(0)
        tgt_len = self.all_attention_scores[0].size(1)
        src_len = self.all_attention_scores[0].size(2)

        # 取第一个样本的最后一个目标token的注意力
        impact_scores = []
        for i, scores in enumerate(self.all_attention_scores):
            # 取第一个样本的最后一个目标token对所有外部变量的注意力
            impact_scores.append(scores[0, -1, :].cpu().numpy())

        # 转换为numpy数组 [n_layers, n_external_vars]
        impact_scores = np.array(impact_scores)

        # 根据注意力矩阵形状生成外部变量名称
        external_var_names = [f"Var{i}" for i in range(impact_scores.shape[1])]

        # 绘制热力图
        plt.figure(figsize=(10, 6))
        im = plt.imshow(impact_scores, cmap="viridis", aspect="auto")

        # 设置坐标轴
        plt.xticks(range(len(external_var_names)), external_var_names, rotation=45)
        plt.yticks(range(self.n_layers), [f"Layer {i+1}" for i in range(self.n_layers)])

        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label("Attention Score")

        # 添加标题
        plt.title("External Variable Impact Across Layers")
        plt.tight_layout()

        # 保存或显示图片
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
