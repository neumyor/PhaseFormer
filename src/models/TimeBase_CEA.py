import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.pl_bases.default_module import DefaultPLModule



class CircularConv1d(nn.Module):
    """循环卷积层，实现列维度的循环等变性"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super(CircularConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=0, dilation=dilation, groups=groups)
    
    def forward(self, x):
        # x: (batch, channels, length)
        # 计算循环填充大小
        pad_size = (self.kernel_size - 1) * self.dilation
        # 循环填充
        x_padded = F.pad(x, (pad_size, 0), mode='circular')
        return self.conv(x_padded)


class CircularRelativePositionBias(nn.Module):
    """环形相对位置偏置"""
    def __init__(self, num_positions, num_heads=1):
        super(CircularRelativePositionBias, self).__init__()
        self.num_positions = num_positions
        self.num_heads = num_heads
        # 只需要存储num_positions个偏置值，对应不同的环形距离
        self.bias_table = nn.Parameter(torch.zeros(num_positions, num_heads))
        
    def forward(self, seq_len):
        # 生成环形相对位置矩阵
        positions = torch.arange(seq_len, device=self.bias_table.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
        # 计算环形距离：(j-i) mod num_positions
        circular_distances = relative_positions % self.num_positions
        # 查表获取偏置
        bias = self.bias_table[circular_distances]  # (seq_len, seq_len, num_heads)
        return bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)


class CircularEquivariantEncoder(nn.Module):
    """循环等变编码器"""
    def __init__(self, period_len, latent_dim, hidden_dims=None):
        super(CircularEquivariantEncoder, self).__init__()
        self.period_len = period_len
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [32, 64]
        self.hidden_dims = hidden_dims
        
        # 修复的特征提取器 - 保留更多空间信息
        self.feature_extractor = nn.Sequential(
            CircularConv1d(1, hidden_dims[0], kernel_size=3),
            nn.GELU(),
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(period_len // 4)  # 保留4-6个特征点而不是1个
        )
        self.fc_compress = nn.Linear(hidden_dims[1] * (period_len // 4), latent_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.feature_extractor(x)  # (batch, hidden_dims[1], period_len//4)
        x_flat = x.flatten(1)  # (batch, hidden_dims[1] * period_len//4)
        latent = self.fc_compress(x_flat)
        return latent


class CircularEquivariantDecoder(nn.Module):
    """循环等变解码器"""
    def __init__(self, latent_dim, period_len, hidden_dims=None):
        super(CircularEquivariantDecoder, self).__init__()
        self.period_len = period_len
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.hidden_dims = hidden_dims
        
        self.expand = nn.Linear(latent_dim, hidden_dims[0] * (period_len // 4))
        self.upsample_conv = nn.Sequential(
            nn.Unflatten(1, (hidden_dims[0], period_len // 4)),
            nn.ConvTranspose1d(hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1),  # 更好的上采样
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dims[1], hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dims[1], 1, kernel_size=3, padding=1)
        )
        
    def forward(self, latent):
        x = self.expand(latent)  # (batch, hidden_dims[0] * period_len//4)
        x = self.upsample_conv(x)  # (batch, 1, period_len)
        return x.squeeze(1)  # (batch, period_len)


class PositionSensitivePredictor(nn.Module):
    """时间位置敏感预测器"""
    def __init__(self, num_periods_input, num_periods_output, latent_dim, mapping_mode="shared"):
        super(PositionSensitivePredictor, self).__init__()
        self.num_periods_input = num_periods_input
        self.num_periods_output = num_periods_output
        self.latent_dim = latent_dim
        self.mapping_mode = mapping_mode  # "shared" 或 "individual"
        
        if mapping_mode == "shared":
            # Shared模式：所有潜在维度共享一个线性层
            self.shared_time_mapper = nn.Linear(num_periods_input, num_periods_output)
        elif mapping_mode == "individual":
            # Individual模式：每个潜在维度有自己独立的线性层
            self.individual_time_mappers = nn.ModuleList([
                nn.Linear(num_periods_input, num_periods_output)
                for _ in range(latent_dim)
            ])
        else:
            raise ValueError(f"不支持的映射模式: {mapping_mode}")
    
    def forward(self, encoded_coeffs):
        """
        encoded_coeffs: (batch, enc_in, num_periods_input, latent_dim)
        输出: (batch, enc_in, num_periods_output, latent_dim)
        """
        batch_size, enc_in = encoded_coeffs.size(0), encoded_coeffs.size(1)
        
        if self.mapping_mode == "shared":
            # Shared模式：所有潜在维度共享一个线性层
            # 重塑为: (batch*enc_in*latent_dim, num_periods_input)
            coeffs_reshaped = encoded_coeffs.reshape(batch_size * enc_in * self.latent_dim, self.num_periods_input)
            # 使用共享的时间映射器
            future_series_flat = self.shared_time_mapper(coeffs_reshaped)
            # 恢复形状: (batch, enc_in, num_periods_output, latent_dim)
            future_coeffs = future_series_flat.reshape(batch_size, enc_in, self.num_periods_output, self.latent_dim)
            
        elif self.mapping_mode == "individual":
            # Individual模式：每个潜在维度有自己独立的线性层
            future_series_list = []
            
            # 对每个潜在维度使用独立的映射器
            for latent_idx in range(self.latent_dim):
                # 提取特定潜在维度的时间序列: (batch*enc_in, num_periods_input)
                latent_series = encoded_coeffs[:, :, :, latent_idx].reshape(batch_size * enc_in, self.num_periods_input)
                # 使用该潜在维度对应的独立映射器
                future_series = self.individual_time_mappers[latent_idx](latent_series)
                future_series_list.append(future_series.reshape(batch_size, enc_in, self.num_periods_output, 1))
            
            # 重新组合: (batch, enc_in, num_periods_output, latent_dim)
            future_coeffs = torch.cat(future_series_list, dim=-1)
        
        return future_coeffs


class TimeBase_CEA(DefaultPLModule):
    """TimeBase循环等变自编码器模型"""
    def __init__(self, configs):
        super(TimeBase_CEA, self).__init__(configs)
        
        # 基础参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        # 强制使用Shared模式
        self.individual = False
        
        # 模型参数
        self.latent_dim = getattr(configs, "latent_dim", 8)
        self.encoder_hidden_dims = getattr(configs, "encoder_hidden_dims", [32, 64])
        self.decoder_hidden_dims = getattr(configs, "decoder_hidden_dims", [64, 32])
        
        # 等变性参数
        self.use_equivariance_regularization = getattr(configs, "use_equivariance_regularization", False)
        self.equivariance_weight = getattr(configs, "equivariance_weight", 0.1)
        
        # 位置敏感预测器参数
        self.use_position_sensitive_predictor = getattr(configs, "use_position_sensitive_predictor", True)
        self.mapping_mode = getattr(configs, "mapping_mode", "shared")  # "shared" 或 "individual"
        
        # 计算周期数
        self.num_periods_input = self.seq_len // self.period_len
        self.num_periods_output = self.pred_len // self.period_len
        
        self.pad_seq_len = 0
        if self.seq_len % self.period_len != 0:
            self.num_periods_input += 1
            self.pad_seq_len = self.num_periods_input * self.period_len - self.seq_len
        
        if self.pred_len % self.period_len != 0:
            self.num_periods_output += 1
        
        # 初始化组件
        self._setup_encoders_decoders()
        self._setup_predictor()
        
    def _setup_encoders_decoders(self):
        """设置编码器和解码器 - 仅Shared模式"""
        self.encoder = CircularEquivariantEncoder(self.period_len, self.latent_dim, self.encoder_hidden_dims)
        self.decoder = CircularEquivariantDecoder(self.latent_dim, self.period_len, self.decoder_hidden_dims)
    
    def _setup_predictor(self):
        """设置预测器 - 仅位置敏感预测器"""
        if self.use_position_sensitive_predictor:
            # 使用位置敏感预测器
            self.predictor = PositionSensitivePredictor(
                num_periods_input=self.num_periods_input,
                num_periods_output=self.num_periods_output,
                latent_dim=self.latent_dim,
                mapping_mode=self.mapping_mode
            )
        else:
            raise ValueError("线性预测器已被删除，请使用位置敏感预测器")
    
    def _encode_periods(self, x):
        """编码所有周期 - 仅Shared模式"""
        batch_size = x.size(0)
        
        # Shared模式：所有变量共享编码器
        x_reshaped = x.reshape(batch_size * self.enc_in * self.num_periods_input, self.period_len)
        coeffs_flat = self.encoder(x_reshaped)
        encoded_coeffs = coeffs_flat.reshape(batch_size, self.enc_in, self.num_periods_input, self.latent_dim)
        
        return encoded_coeffs
    
    def _predict_future_coeffs(self, encoded_coeffs):
        """预测未来系数 - 仅位置敏感预测器"""
        batch_size = encoded_coeffs.size(0)
        
        # 使用位置敏感预测器
        future_coeffs = self.predictor(encoded_coeffs)
        
        return future_coeffs
    
    def _decode_periods(self, coeffs):
        """解码系数为时间序列 - 仅Shared模式"""
        batch_size = coeffs.size(0)
        num_periods = coeffs.size(2)
        
        # Shared模式：所有变量共享解码器
        coeffs_reshaped = coeffs.reshape(batch_size * self.enc_in * num_periods, self.latent_dim)
        periods_flat = self.decoder(coeffs_reshaped)
        decoded_periods = periods_flat.reshape(batch_size, self.enc_in, num_periods, self.period_len)
        
        return decoded_periods
    
    def _compute_equivariance_loss(self, x, encoded_coeffs):
        """计算等变正则化损失 - 仅Shared模式"""
        if not self.use_equivariance_regularization:
            return torch.tensor(0.0, device=x.device)
        
        # 简化版等变损失：只对少量样本计算
        batch_size = x.size(0)
        num_samples = min(batch_size, 2)
        
        total_loss = 0.0
        for sample_idx in range(num_samples):
            # 随机选择一个变量和周期
            var_idx = torch.randint(0, self.enc_in, (1,)).item()
            period_idx = torch.randint(0, self.num_periods_input, (1,)).item()
            
            original_data = x[sample_idx, var_idx, period_idx, :].unsqueeze(0)
            shifted_data = torch.roll(original_data, shifts=1, dims=-1)
            
            # Shared模式：使用共享编码器
            original_encoded = self.encoder(original_data)
            shifted_encoded = self.encoder(shifted_data)
            
            total_loss += F.mse_loss(original_encoded, shifted_encoded)
        
        return total_loss / num_samples

    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """前向传播"""
        x = x_enc
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        
        # 填充处理
        if self.pad_seq_len > 0:
            pad_start = (self.num_periods_input - 1) * self.period_len
            x = torch.cat([x, x[:, :, pad_start - self.pad_seq_len:pad_start]], dim=-1)
        
        # 重塑为周期格式
        x = x.reshape(batch_size, self.enc_in, self.num_periods_input, self.period_len)
        
        # 编码-预测-解码
        encoded_coeffs = self._encode_periods(x)
        future_coeffs = self._predict_future_coeffs(encoded_coeffs)
        decoded_periods = self._decode_periods(future_coeffs)
        
        # 输出处理
        output = decoded_periods.reshape(batch_size, self.enc_in, -1)
        output = output[:, :, :self.pred_len]
        output = output.permute(0, 2, 1)
        
        return output, encoded_coeffs, future_coeffs
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        
        dec_inp = self._build_decoder_input(batch_y)
        outputs, encoded_coeffs, future_coeffs = self(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        outputs = outputs[:, -self.args.dataset_args.pred_len:, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len:, :]
        
        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)
        
        criterion = self._get_criterion(self.args.training_args.loss_func)
        main_loss = criterion(outputs, batch_y)
        
        # 等变损失
        if self.use_equivariance_regularization:
            x_reshaped = batch_x.permute(0, 2, 1)
            if self.pad_seq_len > 0:
                pad_start = (self.num_periods_input - 1) * self.period_len
                x_reshaped = torch.cat([x_reshaped, x_reshaped[:, :, pad_start - self.pad_seq_len:pad_start]], dim=-1)
            x_reshaped = x_reshaped.reshape(batch_x.size(0), self.enc_in, self.num_periods_input, self.period_len)
            equivariance_loss = self._compute_equivariance_loss(x_reshaped, encoded_coeffs)
            total_loss = main_loss + self.equivariance_weight * equivariance_loss
            self.log("train_loss_equivariance", equivariance_loss, on_epoch=True)
        else:
            total_loss = main_loss
        
        self.log("train_loss", total_loss, on_epoch=True)
        return total_loss
    
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
        
        outputs = outputs[:, -self.args.dataset_args.pred_len:, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len:, :]
        
        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)
        
        criterion = self._get_criterion(self.args.training_args.loss_func)
        loss = criterion(outputs, batch_y)
        self.log("val_loss", loss, on_epoch=True)
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
        
        outputs = outputs[:, -self.args.dataset_args.pred_len:, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len:, :]
        
        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)
        
        pred = outputs.detach()
        true = batch_y.detach()
        
        from src.utils.metrics import metric
        loss = metric(pred, true)
        self.log_dict({f"test_{k}": v for k, v in loss.items()}, on_epoch=True)
        return loss
