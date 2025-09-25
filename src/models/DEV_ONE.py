import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from easydict import EasyDict
import numpy as np
from scipy import stats
import torch.distributions as D


class GaussianCopulaTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # 添加小的epsilon值用于数值稳定性
        self.epsilon = 1e-6

    def forward(self, x):
        """
        将输入数据通过Gaussian Copula转换为正态分布
        x: [B, L, D]
        """
        # 边际分布转换 - 计算经验累积分布函数
        B, L, D = x.shape
        x_sorted, _ = torch.sort(x, dim=1)

        # 计算每个值的经验累积概率，改进计算方法
        u = torch.zeros_like(x)
        for i in range(D):
            # 计算每个样本点的累积概率
            ranks = (
                (x[..., i].unsqueeze(1) >= x_sorted[..., i].unsqueeze(2))
                .sum(dim=1)
                .float()
            )
            # 使用更稳定的经验累积分布函数计算方法
            u[..., i] = (ranks + 0.5) / L

            # 确保所有概率值严格位于(0,1)区间内，避免erfinv计算NaN
            u[..., i] = torch.clamp(u[..., i], self.epsilon, 1.0 - self.epsilon)

        # 正态分位数转换，添加数值稳定性处理
        z = torch.erfinv(2 * u - 1) * np.sqrt(2)

        # 计算相关系数矩阵
        z_centered = z - z.mean(dim=1, keepdim=True)
        cov = torch.bmm(z_centered.transpose(1, 2), z_centered) / (L - 1)
        std_devs = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).unsqueeze(2))

        # 添加小的epsilon值避免除零错误
        corr_matrix = cov / (torch.bmm(std_devs, std_devs.transpose(1, 2)) + 1e-8)

        return z, corr_matrix


class FactorLoadInference(nn.Module):
    def __init__(self, input_dim, factor_dim):
        super().__init__()
        self.factor_dim = factor_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, 64),  # 输入维度应为 2D（均值+标准差）
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, factor_dim * input_dim),  # 输出维度为 D*m
        )

    def forward(self, x):
        """
        推理因子载荷矩阵
        x: [B, L, D]
        """
        # 使用序列的统计特征作为输入
        x_mean = x.mean(dim=1)  # [B, D]
        x_std = x.std(dim=1)  # [B, D]
        x_stats = torch.cat([x_mean, x_std], dim=1)  # [B, 2D]

        # 输出因子载荷矩阵 [B, D, m]
        batch_size, original_input_dim = x_mean.shape  # 原始输入维度 D
        lambda_matrix = self.mlp(x_stats).view(
            batch_size, original_input_dim, self.factor_dim
        )

        # 确保载荷矩阵半正定
        lambda_matrix = F.softplus(lambda_matrix)

        return lambda_matrix


class FactorLoadInference(nn.Module):
    def __init__(self, input_dim, factor_dim):
        super().__init__()
        self.factor_dim = factor_dim
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, factor_dim * input_dim),  # 输出下三角矩阵元素
        )

    def forward(self, x):
        # 使用序列的统计特征作为输入
        x_mean = x.mean(dim=1)  # [B, D]
        x_std = x.std(dim=1)  # [B, D]
        x_stats = torch.cat([x_mean, x_std], dim=1)  # [B, 2D]

        # 输出因子载荷矩阵 [B, D, m]
        batch_size, original_input_dim = x_mean.shape  # 原始输入维度 D
        lambda_lower = self.mlp(x_stats).view(
            batch_size, self.input_dim, self.factor_dim
        )
        # 通过 Cholesky 分解构建半正定矩阵
        lambda_matrix = torch.bmm(lambda_lower, lambda_lower.transpose(1, 2))
        return lambda_matrix


class SharedBackbone(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len  # 显式保存序列长度
        self.embedding = nn.Linear(seq_len, d_model)  # 输入维度为 seq_len（L）
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout
            ),
            n_layers,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, L, D] -> [B, D, L] -> 展平为 [B*D, L] -> 嵌入 -> [B*D, d_model] -> 恢复为 [B, D, d_model]
        """
        B, L, D = x.shape  # 获取批次、序列长度、特征维度
        x = x.transpose(1, 2)  # [B, D, L]
        x = x.reshape(B * D, L)  # 展平为二维张量 [B*D, L]
        x = self.embedding(x)  # [B*D, d_model]
        x = x.reshape(B, D, self.embedding.out_features)  # 恢复为 [B, D, d_model]
        x = x.permute(1, 0, 2)  # [D, B, d_model]（Transformer要求序列维度在前）
        x = self.transformer(x)  # [D, B, d_model]
        x = x.permute(1, 0, 2)  # [B, D, d_model]
        x = self.norm(x)
        return x


class IndependentPredictionHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.factor_dim = configs.factor_dim
        self.feature_dim = configs.feature_dim

        # 隐变量系数 - 每个变量一个
        self.u = nn.Parameter(torch.randn(configs.feature_dim, configs.factor_dim))

        # 偏置项 - 变量特异性静态偏移
        self.bias = nn.Parameter(torch.zeros(configs.feature_dim))

        # 独特因子方差（每个变量一个独立参数）
        self.psi = nn.Parameter(torch.randn(configs.feature_dim))  # 初始化为随机噪声

        # 添加线性层将 d_model 转换为 factor_dim
        self.proj = nn.Linear(configs.d_model, configs.factor_dim)

    @property
    def psi_positive(self):
        """确保独特因子方差非负"""
        return F.softplus(self.psi)  # 使用softplus激活函数保证非负性

    def forward(self, h, lambda_matrix):
        """
        h: [B, D, d_model]
        lambda_matrix: [B, D, m]
        self.u: [D, m] -> 转置为 [m, D]
        """
        batch_size, feature_dim, _ = h.shape  # feature_dim = D
        h = self.proj(h)  # [B, D, d_model] → [B, D, factor_dim]
        u_T = self.u.transpose(0, 1)  # [m, D]
        u_expanded = u_T.unsqueeze(0).expand(batch_size, -1, -1)  # [B, m, D]
        w = torch.bmm(lambda_matrix, u_expanded)  # [B, D, m] @ [B, m, D] = [B, D, D]
        y = torch.bmm(h, w)  # [B, D, m] @ [B, m, D] = [B, D, D]
        y = torch.diagonal(y, dim1=1, dim2=2).unsqueeze(2)  # [B, D, 1]
        y = y + self.bias.unsqueeze(0).unsqueeze(2)  # 添加偏置
        
        return y.expand(-1, -1, self.pred_len)  # [B, D, pred_len]


class DEV_ONE(nn.Module):
    def __init__(self, configs: EasyDict):
        """
        configs: dict, 模型配置参数
        """
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.feature_dim = configs.feature_dim
        self.factor_dim = configs.factor_dim
        self.d_model = configs.d_model

        # 数据标准化模块
        self.copula_transform = GaussianCopulaTransform()

        # 因子载荷推理模型
        self.factor_load_inference = FactorLoadInference(
            configs.feature_dim, configs.factor_dim
        )

        # 共享骨干网络
        self.backbone = SharedBackbone(
            configs.seq_len,
            configs.d_model,
            configs.n_heads,
            configs.n_layers,
            configs.dropout,
        )

        # 独立预测头
        self.prediction_head = IndependentPredictionHead(configs)

    def forward(self, x_enc: torch.Tensor):
        """
        x_enc: [B, L, D], 输入多元时间序列
        return: [B, D, pred_len], 输出多元时间序列预测
        """
        batch_size, seq_len, feature_dim = x_enc.shape

        # 1. 数据标准化
        z, corr_matrix = self.copula_transform(x_enc)

        # 2. 因子载荷推理
        lambda_matrix = self.factor_load_inference(z)

        # 3. 共享骨干网络特征提取
        h = self.backbone(z)

        # 4. 独立预测头
        y_pred = self.prediction_head(h, lambda_matrix)

        return y_pred, lambda_matrix, self.prediction_head.u

    def compute_covariance(self, lambda_matrix, u):
        """计算包含公共因子和独特因子的协方差矩阵"""
        batch_size = lambda_matrix.size(0)
        lambda_lambda_t = torch.bmm(
            lambda_matrix, lambda_matrix.transpose(1, 2)
        )  # [B, D, D] 公共因子协方差
        u_expanded = u.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, D, m] 扩展隐变量系数

        # 公共因子部分：u^⊤ (ΛΛ^⊤) u
        common_cov = torch.bmm(
            torch.bmm(u_expanded, lambda_lambda_t), u_expanded.transpose(1, 2)
        )  # [B, D, D]

        # 独特因子部分：对角矩阵 diag(ψ)
        psi_diag = (
            torch.diag_embed(self.prediction_head.psi_positive)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )  # [B, D, D]

        # 总协方差矩阵
        total_cov = common_cov + psi_diag  # 公共因子 + 独特因子
        return total_cov

    def compute_regularization(self, lambda_matrix, u, w_t=None):
        """计算时序协同正则项"""
        batch_size, _, feature_dim, _ = lambda_matrix.shape

        # 计算ΛΛ^T
        lambda_lambda_t = torch.bmm(
            lambda_matrix, lambda_matrix.transpose(1, 2)
        )  # [B, D, D]

        # 计算隐变量矩阵 U = [u_1, u_2, ..., u_D]^T
        U = u.T  # [D, m]

        # 构建差分矩阵 Δ
        eye = torch.eye(feature_dim, device=lambda_matrix.device)
        delta = eye.unsqueeze(0) - eye.unsqueeze(1)  # [D, D, D]
        delta = delta.reshape(-1, feature_dim)  # [D*D, D]

        if w_t is None:
            # 如果没有提供动态权重，使用ΛΛ^T
            W = lambda_lambda_t
        else:
            # 或者使用所有时间步的平均权重作为静态权重
            W = w_t.mean(
                dim=1
            ).squeeze()  # [B, D, D] -> [B, D, D]（取平均）或 [D, D]（如果batch=1）

        # 处理批量维度
        if W.dim() == 2:
            W = W.expand(batch_size, -1, -1)  # [D, D] -> [B, D, D]

        # 计算 W 向量化后的对角矩阵 diag(W_vec)
        W_vec = W.reshape(batch_size, -1)  # [B, D*D]
        diag_W_vec = torch.diag_embed(W_vec)  # [B, D*D, D*D]

        # 计算中间矩阵 W_hat = Δ^T · diag(W_vec) · Δ
        delta_t = delta.t()  # [D, D*D]
        W_hat = torch.bmm(
            torch.bmm(delta_t.expand(batch_size, -1, -1), diag_W_vec),
            delta.expand(batch_size, -1, -1),
        )  # [B, D, D]

        # 计算 U·W_hat·U^T
        UWU = torch.bmm(
            torch.bmm(U.expand(batch_size, -1, -1), W_hat),
            U.expand(batch_size, -1, -1).transpose(1, 2),
        )  # [B, D, D]

        # 计算迹 tr(UWU·ΛΛ^T)
        trace = (
            torch.bmm(UWU, lambda_lambda_t).diagonal(dim1=1, dim2=2).sum(dim=1)
        )  # [B]

        # 返回正则化项的平均值
        return trace.mean()

    def get_loss_function(self):
        """
        返回模型训练的损失函数
        """

        def loss_function(y_pred, y_true, lambda_matrix, u, w_t, alpha=0.5, beta=0.1):
            """
            y_pred: [B, D, pred_len] - 模型预测值
            y_true: [B, D, pred_len] - 真实值
            lambda_matrix: [B, D, m] - 因子载荷矩阵
            u: [D, m] - 隐变量系数
            w_t: [B, L, D, D] - 时变权重矩阵
            alpha: float - 正则化项权重
            beta: float - 协方差约束权重
            """
            # 1. 计算预测误差
            reconstruction_error = F.mse_loss(y_pred, y_true)

            # 2. 计算时序协同正则项
            regularization = self.compute_regularization(lambda_matrix, u, w_t)

            # 计算包含独特因子的预测协方差
            pred_cov = self.compute_covariance(lambda_matrix, u)

            # 经验协方差（基于真实值）
            y_true_centered = y_true - y_true.mean(dim=2, keepdim=True)
            emp_cov = torch.bmm(y_true_centered, y_true_centered.transpose(1, 2)) / (
                y_true.size(2) - 1
            )

            # 协方差约束损失（包含独特因子）
            cov_loss = F.mse_loss(pred_cov, emp_cov)

            total_loss = reconstruction_error + alpha * regularization + beta * cov_loss
            return total_loss

        return loss_function