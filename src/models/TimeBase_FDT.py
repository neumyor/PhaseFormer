import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.pl_bases.default_module import DefaultPLModule


def cal_orthogonal_loss(matrix):
    gram_matrix = torch.matmul(matrix.transpose(-2, -1), matrix)
    one_diag = torch.diagonal(gram_matrix, dim1=-2, dim2=-1)
    two_diag = torch.diag_embed(one_diag)
    off_diagonal = gram_matrix - two_diag
    loss = torch.norm(off_diagonal, dim=(-2, -1))
    return loss.mean()


class LearnableFourierLayer(nn.Module):
    """可学习的傅里叶变换层"""
    def __init__(self, period_len, num_frequencies, learnable=True):
        super(LearnableFourierLayer, self).__init__()
        self.period_len = period_len
        self.num_frequencies = num_frequencies
        self.learnable = learnable
        
        if learnable:
            # 可学习的傅里叶字典：初始化为标准DFT基，允许微调
            # 频率偏移参数
            self.frequency_offsets = nn.Parameter(torch.zeros(num_frequencies))
            # 幅度权重参数
            self.amplitude_weights = nn.Parameter(torch.ones(num_frequencies))
            # 相位偏移参数
            self.phase_offsets = nn.Parameter(torch.zeros(num_frequencies))
        else:
            # 固定DFT
            self.register_buffer('frequency_indices', torch.arange(num_frequencies))
    
    def _get_dft_matrix(self, seq_len, device):
        """获取DFT变换矩阵"""
        if self.learnable:
            # 可学习版本
            frequencies = torch.arange(self.num_frequencies, device=device).float() + self.frequency_offsets
            amplitudes = self.amplitude_weights
            phases = self.phase_offsets
        else:
            # 固定版本
            frequencies = self.frequency_indices.float()
            amplitudes = torch.ones(self.num_frequencies, device=device)
            phases = torch.zeros(self.num_frequencies, device=device)
        
        # 构建DFT矩阵
        n = torch.arange(seq_len, device=device).float().unsqueeze(0)  # (1, seq_len)
        k = frequencies.unsqueeze(1)  # (num_frequencies, 1)
        
        # 计算复数指数 exp(-2πi * k * n / N)
        angles = -2 * math.pi * k * n / seq_len + phases.unsqueeze(1)
        
        # 实部和虚部
        real_part = amplitudes.unsqueeze(1) * torch.cos(angles)  # (num_frequencies, seq_len)
        imag_part = amplitudes.unsqueeze(1) * torch.sin(angles)  # (num_frequencies, seq_len)
        
        return real_part, imag_part
    
    def forward(self, x):
        """
        前向传播：将时域信号转换为频域系数
        x: (batch, seq_len)
        return: (batch, num_frequencies, 2) - 最后一维是[实部, 虚部]
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # 获取DFT矩阵
        dft_real, dft_imag = self._get_dft_matrix(seq_len, device)
        
        # 计算频域系数
        real_coeffs = torch.matmul(dft_real, x.transpose(-2, -1))  # (num_frequencies, batch)
        imag_coeffs = torch.matmul(dft_imag, x.transpose(-2, -1))  # (num_frequencies, batch)
        
        # 重新排列维度
        real_coeffs = real_coeffs.transpose(0, 1)  # (batch, num_frequencies)
        imag_coeffs = imag_coeffs.transpose(0, 1)  # (batch, num_frequencies)
        
        # 合并实部和虚部
        complex_coeffs = torch.stack([real_coeffs, imag_coeffs], dim=-1)  # (batch, num_frequencies, 2)
        
        return complex_coeffs


class LearnableInverseFourierLayer(nn.Module):
    """可学习的逆傅里叶变换层"""
    def __init__(self, period_len, num_frequencies, learnable=True):
        super(LearnableInverseFourierLayer, self).__init__()
        self.period_len = period_len
        self.num_frequencies = num_frequencies
        self.learnable = learnable
        
        if learnable:
            # 与前向变换共享参数（通过引用传递）
            self.frequency_offsets = nn.Parameter(torch.zeros(num_frequencies))
            self.amplitude_weights = nn.Parameter(torch.ones(num_frequencies))
            self.phase_offsets = nn.Parameter(torch.zeros(num_frequencies))
        else:
            self.register_buffer('frequency_indices', torch.arange(num_frequencies))
    
    def _get_idft_matrix(self, seq_len, device):
        """获取逆DFT变换矩阵"""
        if self.learnable:
            frequencies = torch.arange(self.num_frequencies, device=device).float() + self.frequency_offsets
            amplitudes = self.amplitude_weights
            phases = self.phase_offsets
        else:
            frequencies = self.frequency_indices.float()
            amplitudes = torch.ones(self.num_frequencies, device=device)
            phases = torch.zeros(self.num_frequencies, device=device)
        
        # 构建逆DFT矩阵
        n = torch.arange(seq_len, device=device).float().unsqueeze(1)  # (seq_len, 1)
        k = frequencies.unsqueeze(0)  # (1, num_frequencies)
        
        # 计算复数指数 exp(2πi * k * n / N) (注意符号相反)
        angles = 2 * math.pi * k * n / seq_len + phases.unsqueeze(0)
        
        # 实部和虚部矩阵
        real_part = amplitudes.unsqueeze(0) * torch.cos(angles) / seq_len  # (seq_len, num_frequencies)
        imag_part = amplitudes.unsqueeze(0) * torch.sin(angles) / seq_len  # (seq_len, num_frequencies)
        
        return real_part, imag_part
    
    def forward(self, complex_coeffs):
        """
        逆变换：将频域系数转换回时域信号
        complex_coeffs: (batch, num_frequencies, 2) - 最后一维是[实部, 虚部]
        return: (batch, seq_len)
        """
        batch_size = complex_coeffs.size(0)
        device = complex_coeffs.device
        
        # 分离实部和虚部
        real_coeffs = complex_coeffs[:, :, 0]  # (batch, num_frequencies)
        imag_coeffs = complex_coeffs[:, :, 1]  # (batch, num_frequencies)
        
        # 获取逆DFT矩阵
        idft_real, idft_imag = self._get_idft_matrix(self.period_len, device)
        
        # 计算时域信号：实部
        time_signal = torch.matmul(real_coeffs, idft_real.transpose(0, 1)) - torch.matmul(imag_coeffs, idft_imag.transpose(0, 1))
        
        return time_signal


class ComplexGRU(nn.Module):
    """复数GRU用于频域系数的时序预测"""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(ComplexGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 将复数分解为实部和虚部，输入维度翻倍
        self.real_input_size = input_size * 2  # [实部, 虚部]
        self.gru = nn.GRU(self.real_input_size, hidden_size, num_layers, batch_first=True)
        
        # 输出层：预测复数系数
        self.output_real = nn.Linear(hidden_size, input_size)
        self.output_imag = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, steps=1):
        """
        前向传播
        x: (batch, seq_len, num_frequencies, 2) - 最后一维是[实部, 虚部]
        steps: 预测的步数
        return: (batch, steps, num_frequencies, 2)
        """
        batch_size, seq_len, num_frequencies, _ = x.shape
        
        # 重塑输入：将复数展平为实数向量
        x_flat = x.reshape(batch_size, seq_len, -1)  # (batch, seq_len, num_frequencies*2)
        
        # GRU前向传播
        gru_out, hidden = self.gru(x_flat)  # gru_out: (batch, seq_len, hidden_size)
        
        # 使用最后一个隐状态进行多步预测
        predictions = []
        current_hidden = hidden
        
        for step in range(steps):
            # 预测下一步
            step_real = self.output_real(current_hidden[-1])  # (batch, num_frequencies)
            step_imag = self.output_imag(current_hidden[-1])  # (batch, num_frequencies)
            
            # 合并实部和虚部
            step_complex = torch.stack([step_real, step_imag], dim=-1)  # (batch, num_frequencies, 2)
            predictions.append(step_complex)
            
            # 更新隐状态（使用预测结果作为下一步输入）
            next_input = step_complex.reshape(batch_size, 1, -1)  # (batch, 1, num_frequencies*2)
            _, current_hidden = self.gru(next_input, current_hidden)
        
        # 堆叠所有预测步
        predictions = torch.stack(predictions, dim=1)  # (batch, steps, num_frequencies, 2)
        
        return predictions


class ComplexTransformer(nn.Module):
    """复数Transformer用于频域系数的时序预测"""
    def __init__(self, num_frequencies, d_model, nhead, num_layers, seq_len, pred_len):
        super(ComplexTransformer, self).__init__()
        self.num_frequencies = num_frequencies
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 输入投影：将复数系数映射到d_model维度
        self.input_projection = nn.Linear(num_frequencies * 2, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 多步预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len * num_frequencies * 2),
        )
    
    def _generate_causal_mask(self, seq_len):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()
    
    def forward(self, x):
        """
        前向传播
        x: (batch, seq_len, num_frequencies, 2)
        return: (batch, pred_len, num_frequencies, 2)
        """
        batch_size, seq_len, num_frequencies, _ = x.shape
        
        # 重塑并投影到模型维度
        x_flat = x.reshape(batch_size, seq_len, -1)  # (batch, seq_len, num_frequencies*2)
        x_proj = self.input_projection(x_flat)  # (batch, seq_len, d_model)
        
        # 添加位置编码
        x_proj = x_proj + self.pos_encoding[:, :seq_len, :]
        
        # 生成因果掩码
        causal_mask = self._generate_causal_mask(seq_len).to(x.device)
        
        # Transformer编码
        encoded = self.transformer(x_proj, mask=causal_mask)  # (batch, seq_len, d_model)
        
        # 使用最后一个时间步进行预测
        last_hidden = encoded[:, -1, :]  # (batch, d_model)
        
        # 多步预测
        pred_flat = self.prediction_head(last_hidden)  # (batch, pred_len * num_frequencies * 2)
        pred = pred_flat.reshape(batch_size, self.pred_len, num_frequencies, 2)
        
        return pred


class TimeBase_FDT(DefaultPLModule):
    """
    TimeBase Frequency Domain Transformer (FDT) 模型：
    可学习频域层 + 复值RNN/Transformer + iDFT解码
    
    策略特点：
    - 使用频域表示保证循环移位等变性
    - 可学习的傅里叶字典增强拟合能力
    - 复值时序建模处理幅度和相位
    - 频域损失稳定相位学习
    - 支持任意输入和输出长度
    
    适用场景：
    - 周期性强且相位漂移明显的时间序列
    - 样本较少但需要稳健预测
    - 信号在频域有稀疏特性
    """
    def __init__(self, configs):
        super(TimeBase_FDT, self).__init__(configs)
        
        # 基础参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.individual = configs.individual
        
        # FDT特有参数
        self.num_frequencies = getattr(configs, "num_frequencies", 6)  # 包含DC分量
        self.learnable_fourier = getattr(configs, "learnable_fourier", True)
        self.use_complex_transformer = getattr(configs, "use_complex_transformer", True)
        
        # Transformer参数
        self.d_model = getattr(configs, "d_model", 64)
        self.nhead = getattr(configs, "nhead", 4)
        self.num_layers = getattr(configs, "num_layers", 2)
        
        # GRU参数（备选）
        self.hidden_size = getattr(configs, "hidden_size", 32)
        self.gru_layers = getattr(configs, "gru_layers", 1)
        
        # 损失权重
        self.magnitude_loss_weight = getattr(configs, "magnitude_loss_weight", 1.0)
        self.phase_loss_weight = getattr(configs, "phase_loss_weight", 0.5)
        self.phase_consistency_weight = getattr(configs, "phase_consistency_weight", 0.2)
        
        # 计算时间步数
        self.num_periods_input = self.seq_len // self.period_len
        self.num_periods_output = self.pred_len // self.period_len
        
        # 处理不完整周期
        self.pad_seq_len = 0
        if self.seq_len % self.period_len != 0:
            self.num_periods_input += 1
            self.pad_seq_len = self.num_periods_input * self.period_len - self.seq_len
        
        if self.pred_len % self.period_len != 0:
            self.num_periods_output += 1
        
        # 初始化模块
        self._setup_fourier_layers()
        self._setup_temporal_predictors()
    
    def _setup_fourier_layers(self):
        """设置傅里叶变换层"""
        if self.individual:
            # Individual模式：每个变量独立的傅里叶层
            self.fourier_encoders = nn.ModuleList([
                LearnableFourierLayer(self.period_len, self.num_frequencies, self.learnable_fourier)
                for _ in range(self.enc_in)
            ])
            self.fourier_decoders = nn.ModuleList([
                LearnableInverseFourierLayer(self.period_len, self.num_frequencies, self.learnable_fourier)
                for _ in range(self.enc_in)
            ])
        else:
            # Shared模式：所有变量共享傅里叶层
            self.fourier_encoder = LearnableFourierLayer(self.period_len, self.num_frequencies, self.learnable_fourier)
            self.fourier_decoder = LearnableInverseFourierLayer(self.period_len, self.num_frequencies, self.learnable_fourier)
    
    def _setup_temporal_predictors(self):
        """设置时序预测器"""
        if self.individual:
            # Individual模式：每个变量独立的预测器
            if self.use_complex_transformer:
                self.temporal_predictors = nn.ModuleList([
                    ComplexTransformer(
                        num_frequencies=self.num_frequencies,
                        d_model=self.d_model,
                        nhead=self.nhead,
                        num_layers=self.num_layers,
                        seq_len=self.num_periods_input,
                        pred_len=self.num_periods_output
                    ) for _ in range(self.enc_in)
                ])
            else:
                self.temporal_predictors = nn.ModuleList([
                    ComplexGRU(
                        input_size=self.num_frequencies,
                        hidden_size=self.hidden_size,
                        num_layers=self.gru_layers
                    ) for _ in range(self.enc_in)
                ])
        else:
            # Shared模式：所有变量共享预测器
            if self.use_complex_transformer:
                self.temporal_predictor = ComplexTransformer(
                    num_frequencies=self.num_frequencies,
                    d_model=self.d_model,
                    nhead=self.nhead,
                    num_layers=self.num_layers,
                    seq_len=self.num_periods_input,
                    pred_len=self.num_periods_output
                )
            else:
                self.temporal_predictor = ComplexGRU(
                    input_size=self.num_frequencies,
                    hidden_size=self.hidden_size,
                    num_layers=self.gru_layers
                )
    
    def _encode_to_frequency_domain(self, x):
        """编码到频域"""
        # x: (batch, enc_in, num_periods, period_len)
        batch_size = x.size(0)
        
        if self.individual:
            # Individual模式
            freq_coeffs = []
            for var_idx in range(self.enc_in):
                var_coeffs = []
                for period_idx in range(self.num_periods_input):
                    period_data = x[:, var_idx, period_idx, :]  # (batch, period_len)
                    coeffs = self.fourier_encoders[var_idx](period_data)  # (batch, num_frequencies, 2)
                    var_coeffs.append(coeffs)
                var_coeffs = torch.stack(var_coeffs, dim=1)  # (batch, num_periods, num_frequencies, 2)
                freq_coeffs.append(var_coeffs)
            freq_coeffs = torch.stack(freq_coeffs, dim=1)  # (batch, enc_in, num_periods, num_frequencies, 2)
        else:
            # Shared模式
            x_reshaped = x.reshape(batch_size * self.enc_in * self.num_periods_input, self.period_len)
            coeffs_flat = self.fourier_encoder(x_reshaped)  # (batch*enc_in*num_periods, num_frequencies, 2)
            freq_coeffs = coeffs_flat.reshape(batch_size, self.enc_in, self.num_periods_input, self.num_frequencies, 2)
        
        return freq_coeffs
    
    def _predict_future_frequencies(self, freq_coeffs):
        """预测未来频域系数"""
        batch_size = freq_coeffs.size(0)
        
        if self.individual:
            # Individual模式
            future_coeffs = []
            for var_idx in range(self.enc_in):
                var_coeffs = freq_coeffs[:, var_idx, :, :, :]  # (batch, num_periods, num_frequencies, 2)
                
                if self.use_complex_transformer:
                    pred_coeffs = self.temporal_predictors[var_idx](var_coeffs)  # (batch, num_periods_output, num_frequencies, 2)
                else:
                    pred_coeffs = self.temporal_predictors[var_idx](var_coeffs, self.num_periods_output)  # (batch, num_periods_output, num_frequencies, 2)
                
                future_coeffs.append(pred_coeffs)
            future_coeffs = torch.stack(future_coeffs, dim=1)  # (batch, enc_in, num_periods_output, num_frequencies, 2)
        else:
            # Shared模式
            coeffs_reshaped = freq_coeffs.reshape(batch_size * self.enc_in, self.num_periods_input, self.num_frequencies, 2)
            
            if self.use_complex_transformer:
                pred_coeffs_flat = self.temporal_predictor(coeffs_reshaped)  # (batch*enc_in, num_periods_output, num_frequencies, 2)
            else:
                pred_coeffs_flat = self.temporal_predictor(coeffs_reshaped, self.num_periods_output)  # (batch*enc_in, num_periods_output, num_frequencies, 2)
            
            future_coeffs = pred_coeffs_flat.reshape(batch_size, self.enc_in, self.num_periods_output, self.num_frequencies, 2)
        
        return future_coeffs
    
    def _decode_from_frequency_domain(self, freq_coeffs):
        """从频域解码"""
        # freq_coeffs: (batch, enc_in, num_periods, num_frequencies, 2)
        batch_size = freq_coeffs.size(0)
        num_periods = freq_coeffs.size(2)
        
        if self.individual:
            # Individual模式
            decoded_periods = []
            for var_idx in range(self.enc_in):
                var_periods = []
                for period_idx in range(num_periods):
                    period_coeffs = freq_coeffs[:, var_idx, period_idx, :, :]  # (batch, num_frequencies, 2)
                    period_data = self.fourier_decoders[var_idx](period_coeffs)  # (batch, period_len)
                    var_periods.append(period_data)
                var_periods = torch.stack(var_periods, dim=1)  # (batch, num_periods, period_len)
                decoded_periods.append(var_periods)
            decoded_periods = torch.stack(decoded_periods, dim=1)  # (batch, enc_in, num_periods, period_len)
        else:
            # Shared模式
            coeffs_reshaped = freq_coeffs.reshape(batch_size * self.enc_in * num_periods, self.num_frequencies, 2)
            periods_flat = self.fourier_decoder(coeffs_reshaped)  # (batch*enc_in*num_periods, period_len)
            decoded_periods = periods_flat.reshape(batch_size, self.enc_in, num_periods, self.period_len)
        
        return decoded_periods
    
    def _compute_magnitude_loss(self, pred_coeffs, target_coeffs):
        """计算幅度损失"""
        # 计算复数的模长
        pred_magnitude = torch.sqrt(pred_coeffs[:, :, :, :, 0]**2 + pred_coeffs[:, :, :, :, 1]**2)
        target_magnitude = torch.sqrt(target_coeffs[:, :, :, :, 0]**2 + target_coeffs[:, :, :, :, 1]**2)
        
        return F.l1_loss(pred_magnitude, target_magnitude)
    
    def _compute_phase_loss(self, pred_coeffs, target_coeffs):
        """计算相位损失（使用wrapped angle difference）"""
        # 计算相位角
        pred_phase = torch.atan2(pred_coeffs[:, :, :, :, 1], pred_coeffs[:, :, :, :, 0])
        target_phase = torch.atan2(target_coeffs[:, :, :, :, 1], target_coeffs[:, :, :, :, 0])
        
        # Wrapped angle difference
        phase_diff = pred_phase - target_phase
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        
        return torch.mean(torch.abs(phase_diff))
    
    def _compute_phase_consistency_loss(self, pred_coeffs):
        """计算相位连续性正则化"""
        # 计算相邻时间步相位的二阶差分
        if pred_coeffs.size(2) < 3:  # 需要至少3个时间步
            return torch.tensor(0.0, device=pred_coeffs.device)
        
        phase = torch.atan2(pred_coeffs[:, :, :, :, 1], pred_coeffs[:, :, :, :, 0])
        
        # 一阶差分
        first_diff = phase[:, :, 1:, :] - phase[:, :, :-1, :]
        first_diff = torch.atan2(torch.sin(first_diff), torch.cos(first_diff))
        
        # 二阶差分
        second_diff = first_diff[:, :, 1:, :] - first_diff[:, :, :-1, :]
        second_diff = torch.atan2(torch.sin(second_diff), torch.cos(second_diff))
        
        return torch.mean(torch.abs(second_diff))
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """前向传播"""
        x = x_enc  # (batch, seq_len, enc_in)
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # (batch, enc_in, seq_len)
        
        # Padding处理
        if self.pad_seq_len > 0:
            pad_start = (self.num_periods_input - 1) * self.period_len
            x = torch.cat([x, x[:, :, pad_start - self.pad_seq_len:pad_start]], dim=-1)
        
        # 重塑为周期格式
        x = x.reshape(batch_size, self.enc_in, self.num_periods_input, self.period_len)
        
        # 编码到频域
        freq_coeffs = self._encode_to_frequency_domain(x)  # (batch, enc_in, num_periods_input, num_frequencies, 2)
        
        # 预测未来频域系数
        future_freq_coeffs = self._predict_future_frequencies(freq_coeffs)  # (batch, enc_in, num_periods_output, num_frequencies, 2)
        
        # 解码回时域
        decoded_periods = self._decode_from_frequency_domain(future_freq_coeffs)  # (batch, enc_in, num_periods_output, period_len)
        
        # 重塑回原始格式
        output = decoded_periods.reshape(batch_size, self.enc_in, -1)  # (batch, enc_in, total_pred_len)
        output = output[:, :, :self.pred_len]  # 截断到目标长度
        output = output.permute(0, 2, 1)  # (batch, pred_len, enc_in)
        
        return output, freq_coeffs, future_freq_coeffs
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)
        
        outputs, input_freq_coeffs, pred_freq_coeffs = self(
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
        
        # 主要预测损失
        main_loss = criterion(outputs, batch_y)
        
        # 计算目标的频域系数用于频域损失
        batch_y_reshaped = batch_y.permute(0, 2, 1)  # (batch, enc_in, pred_len)
        if self.pred_len % self.period_len != 0:
            # 需要padding到完整周期
            pad_len = self.num_periods_output * self.period_len - self.pred_len
            batch_y_reshaped = F.pad(batch_y_reshaped, (0, pad_len), mode='replicate')
        
        batch_y_periods = batch_y_reshaped.reshape(batch_y.size(0), self.enc_in, self.num_periods_output, self.period_len)
        target_freq_coeffs = self._encode_to_frequency_domain(batch_y_periods)
        
        # 频域损失
        magnitude_loss = self._compute_magnitude_loss(pred_freq_coeffs, target_freq_coeffs)
        phase_loss = self._compute_phase_loss(pred_freq_coeffs, target_freq_coeffs)
        phase_consistency_loss = self._compute_phase_consistency_loss(pred_freq_coeffs)
        
        # 总损失
        total_loss = (main_loss + 
                     self.magnitude_loss_weight * magnitude_loss + 
                     self.phase_loss_weight * phase_loss + 
                     self.phase_consistency_weight * phase_consistency_loss)
        
        # 记录损失
        self.log("train_loss_main", main_loss, on_epoch=True)
        self.log("train_loss_magnitude", magnitude_loss, on_epoch=True)
        self.log("train_loss_phase", phase_loss, on_epoch=True)
        self.log("train_loss_phase_consistency", phase_consistency_loss, on_epoch=True)
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
