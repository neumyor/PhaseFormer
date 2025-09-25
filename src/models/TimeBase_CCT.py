import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from src.models.pl_bases.default_module import DefaultPLModule


def cal_orthogonal_loss(matrix):
    gram_matrix = torch.matmul(matrix.transpose(-2, -1), matrix)
    one_diag = torch.diagonal(gram_matrix, dim1=-2, dim2=-1)
    two_diag = torch.diag_embed(one_diag)
    off_diagonal = gram_matrix - two_diag
    loss = torch.norm(off_diagonal, dim=(-2, -1))
    return loss.mean()


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


class CircularConvTranspose1d(nn.Module):
    """循环转置卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CircularConvTranspose1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, 
                                               stride=stride, padding=0, dilation=dilation)
    
    def forward(self, x):
        # 转置卷积
        x = self.conv_transpose(x)
        # 由于转置卷积可能改变序列长度，这里简化处理
        return x


class CircularRelativeAttention(nn.Module):
    """环形相对位置注意力"""
    def __init__(self, d_model, num_heads, period_len):
        super(CircularRelativeAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.period_len = period_len
        self.head_dim = d_model // num_heads
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 环形相对位置偏置
        self.relative_bias = nn.Parameter(torch.zeros(period_len, num_heads))
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # 投影到Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 重排维度: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加环形相对位置偏置
        if seq_len <= self.period_len:
            # 生成环形相对位置矩阵
            positions = torch.arange(seq_len, device=x.device)
            relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
            circular_distances = relative_positions % self.period_len
            
            # 获取偏置
            bias = self.relative_bias[circular_distances]  # (seq_len, seq_len, num_heads)
            bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, seq_len, seq_len)
            scores = scores + bias
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out


class CircularFeatureExtractor(nn.Module):
    """列特征提取器（共享、等变）"""
    def __init__(self, period_len, hidden_channels=64):
        super(CircularFeatureExtractor, self).__init__()
        self.period_len = period_len
        self.hidden_channels = hidden_channels
        
        # 第一层循环卷积
        self.conv1 = CircularConv1d(1, hidden_channels, kernel_size=5)
        self.activation1 = nn.GELU()
        
        # 第二层循环卷积（扩张）
        self.conv2 = CircularConv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=2)
        self.activation2 = nn.GELU()
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
        # 可选的循环注意力
        self.use_attention = True
        if self.use_attention:
            self.circular_attention = CircularRelativeAttention(hidden_channels, num_heads=4, period_len=period_len)
    
    def forward(self, x):
        # x: (batch, period_len) -> (batch, 1, period_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 循环卷积特征提取
        x = self.activation1(self.conv1(x))  # (batch, hidden_channels, period_len)
        x = self.activation2(self.conv2(x))  # (batch, hidden_channels, period_len)
        
        # 转换维度用于层归一化和注意力
        x = x.permute(0, 2, 1)  # (batch, period_len, hidden_channels)
        x = self.layer_norm(x)
        
        # 循环注意力
        if self.use_attention:
            x = self.circular_attention(x)
        
        # 转换回卷积格式
        x = x.permute(0, 2, 1)  # (batch, hidden_channels, period_len)
        
        return x


class CircularConvLSTMCell(nn.Module):
    """循环卷积LSTM单元"""
    def __init__(self, input_channels, hidden_channels, period_len, kernel_size=5):
        super(CircularConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.period_len = period_len
        self.kernel_size = kernel_size
        
        # 输入到隐状态的循环卷积
        self.conv_ih = CircularConv1d(input_channels, 4 * hidden_channels, kernel_size)
        # 隐状态到隐状态的循环卷积
        self.conv_hh = CircularConv1d(hidden_channels, 4 * hidden_channels, kernel_size)
        
    def forward(self, input_tensor, h_cur, c_cur):
        # input_tensor: (batch, input_channels, period_len)
        # h_cur, c_cur: (batch, hidden_channels, period_len)
        
        # 计算门控值
        combined_conv = self.conv_ih(input_tensor) + self.conv_hh(h_cur)
        
        # 分离四个门
        i_gate, f_gate, o_gate, g_gate = torch.split(combined_conv, self.hidden_channels, dim=1)
        
        # 应用激活函数
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)
        
        # 更新细胞状态和隐状态
        c_next = f_gate * c_cur + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)
        
        return h_next, c_next


class CircularConvLSTM(nn.Module):
    """多层循环卷积LSTM"""
    def __init__(self, input_channels, hidden_channels, period_len, num_layers=2, kernel_size=5):
        super(CircularConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.period_len = period_len
        self.num_layers = num_layers
        
        # 创建多层ConvLSTM
        self.conv_lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_channels if i == 0 else hidden_channels
            self.conv_lstm_cells.append(
                CircularConvLSTMCell(input_dim, hidden_channels, period_len, kernel_size)
            )
    
    def forward(self, input_sequence, initial_states=None):
        """
        input_sequence: (batch, seq_len, input_channels, period_len)
        """
        batch_size, seq_len, _, _ = input_sequence.shape
        
        # 初始化隐状态
        if initial_states is None:
            h_states = [torch.zeros(batch_size, self.hidden_channels, self.period_len, 
                                  device=input_sequence.device) for _ in range(self.num_layers)]
            c_states = [torch.zeros(batch_size, self.hidden_channels, self.period_len, 
                                  device=input_sequence.device) for _ in range(self.num_layers)]
        else:
            h_states, c_states = initial_states
        
        # 逐时间步前向传播
        outputs = []
        for t in range(seq_len):
            x = input_sequence[:, t, :, :]  # (batch, input_channels, period_len)
            
            # 通过所有层
            for layer_idx in range(self.num_layers):
                h_states[layer_idx], c_states[layer_idx] = self.conv_lstm_cells[layer_idx](
                    x, h_states[layer_idx], c_states[layer_idx]
                )
                x = h_states[layer_idx]
            
            outputs.append(x)
        
        # 堆叠输出
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_channels, period_len)
        
        return outputs, (h_states, c_states)


class CausalConvTCN(nn.Module):
    """因果扩张卷积TCN（行轴时间建模）"""
    def __init__(self, input_channels, hidden_channels, num_layers=3, kernel_size=3):
        super(CausalConvTCN, self).__init__()
        self.num_layers = num_layers
        
        # 构建扩张卷积层
        self.conv_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            input_dim = input_channels if i == 0 else hidden_channels
            
            # 因果扩张卷积
            conv_layer = nn.Sequential(
                nn.Conv1d(input_dim, hidden_channels, kernel_size, 
                         padding=padding, dilation=dilation),
                nn.GELU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, 
                         padding=padding, dilation=dilation),
                nn.GELU()
            )
            
            self.conv_layers.append(conv_layer)
            
            # 残差连接
            if input_dim != hidden_channels:
                self.residual_layers.append(nn.Conv1d(input_dim, hidden_channels, 1))
            else:
                self.residual_layers.append(nn.Identity())
    
    def forward(self, x):
        # x: (batch, seq_len, channels, period_len)
        # 重塑为 (batch * period_len, channels, seq_len) 进行行轴卷积
        batch_size, seq_len, channels, period_len = x.shape
        
        # 转换维度：每个位置独立处理时间序列
        x_reshaped = x.permute(0, 3, 2, 1).reshape(batch_size * period_len, channels, seq_len)
        
        # 通过TCN层
        for i, (conv_layer, residual_layer) in enumerate(zip(self.conv_layers, self.residual_layers)):
            # 因果卷积（去掉未来信息）
            conv_out = conv_layer(x_reshaped)
            conv_out = conv_out[:, :, :seq_len]  # 截断到原长度（去掉padding的未来信息）
            
            # 残差连接
            residual = residual_layer(x_reshaped)
            x_reshaped = conv_out + residual
        
        # 转换回原始维度
        x_output = x_reshaped.reshape(batch_size, period_len, -1, seq_len).permute(0, 3, 2, 1)
        
        return x_output


class CircularDecoder(nn.Module):
    """循环等变解码器"""
    def __init__(self, input_channels, period_len, num_layers=2):
        super(CircularDecoder, self).__init__()
        self.period_len = period_len
        
        # 解码层
        self.decode_layers = nn.ModuleList()
        channels = [input_channels, input_channels // 2, 1]
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # 最后一层：输出单通道
                self.decode_layers.append(
                    nn.Sequential(
                        CircularConv1d(channels[i], channels[i+1], kernel_size=5),
                        nn.Tanh()  # 限制输出范围
                    )
                )
            else:
                self.decode_layers.append(
                    nn.Sequential(
                        CircularConv1d(channels[i], channels[i+1], kernel_size=5),
                        nn.GELU(),
                        nn.LayerNorm([channels[i+1], period_len])
                    )
                )
    
    def forward(self, x):
        # x: (batch, input_channels, period_len)
        for layer in self.decode_layers:
            x = layer(x)
        
        return x.squeeze(1)  # (batch, period_len)


class TimeBase_CCT(DefaultPLModule):
    """
    TimeBase Circular ConvLSTM/TCN (CCT) 模型：
    Circular ConvLSTM/Conv-TCN端到端两步生成
    
    策略特点：
    - 列轴循环卷积实现等变性
    - ConvLSTM或TCN进行行轴时序建模
    - 端到端训练，无需显式低维系数
    - 支持Teacher Forcing和Scheduled Sampling
    - 处理相位漂移和循环平移
    
    适用场景：
    - 局部模式和相位漂移都重要的时间序列
    - 需要端到端学习的复杂模式
    - 对解释性要求不高但效果优先的场景
    """
    def __init__(self, configs):
        super(TimeBase_CCT, self).__init__(configs)
        
        # 基础参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.individual = configs.individual
        
        # CCT特有参数
        self.hidden_channels = getattr(configs, "hidden_channels", 64)
        self.use_convlstm = getattr(configs, "use_convlstm", True)  # True: ConvLSTM, False: TCN
        self.convlstm_layers = getattr(configs, "convlstm_layers", 2)
        self.tcn_layers = getattr(configs, "tcn_layers", 3)
        
        # Scheduled Sampling参数
        self.use_scheduled_sampling = getattr(configs, "use_scheduled_sampling", True)
        self.scheduled_sampling_ratio = getattr(configs, "scheduled_sampling_ratio", 0.0)  # 初始值
        self.max_scheduled_sampling_ratio = getattr(configs, "max_scheduled_sampling_ratio", 0.5)
        self.scheduled_sampling_decay_steps = getattr(configs, "scheduled_sampling_decay_steps", 5000)
        
        # 损失权重
        self.spectral_loss_weight = getattr(configs, "spectral_loss_weight", 0.3)
        self.smoothness_loss_weight = getattr(configs, "smoothness_loss_weight", 0.1)
        
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
        
        # 训练步数计数器（用于Scheduled Sampling）
        self.register_buffer('training_step_count', torch.tensor(0))
        
        # 初始化模块
        self._setup_feature_extractors()
        self._setup_temporal_models()
        self._setup_decoders()
    
    def _setup_feature_extractors(self):
        """设置列特征提取器"""
        if self.individual:
            # Individual模式：每个变量独立的特征提取器
            self.feature_extractors = nn.ModuleList([
                CircularFeatureExtractor(self.period_len, self.hidden_channels)
                for _ in range(self.enc_in)
            ])
        else:
            # Shared模式：所有变量共享特征提取器
            self.feature_extractor = CircularFeatureExtractor(self.period_len, self.hidden_channels)
    
    def _setup_temporal_models(self):
        """设置时序建模器"""
        if self.individual:
            # Individual模式：每个变量独立的时序模型
            if self.use_convlstm:
                self.temporal_models = nn.ModuleList([
                    CircularConvLSTM(
                        input_channels=self.hidden_channels,
                        hidden_channels=self.hidden_channels,
                        period_len=self.period_len,
                        num_layers=self.convlstm_layers
                    ) for _ in range(self.enc_in)
                ])
            else:
                self.temporal_models = nn.ModuleList([
                    CausalConvTCN(
                        input_channels=self.hidden_channels,
                        hidden_channels=self.hidden_channels,
                        num_layers=self.tcn_layers
                    ) for _ in range(self.enc_in)
                ])
        else:
            # Shared模式：所有变量共享时序模型
            if self.use_convlstm:
                self.temporal_model = CircularConvLSTM(
                    input_channels=self.hidden_channels,
                    hidden_channels=self.hidden_channels,
                    period_len=self.period_len,
                    num_layers=self.convlstm_layers
                )
            else:
                self.temporal_model = CausalConvTCN(
                    input_channels=self.hidden_channels,
                    hidden_channels=self.hidden_channels,
                    num_layers=self.tcn_layers
                )
    
    def _setup_decoders(self):
        """设置解码器"""
        if self.individual:
            # Individual模式：每个变量独立的解码器
            self.decoders = nn.ModuleList([
                CircularDecoder(self.hidden_channels, self.period_len)
                for _ in range(self.enc_in)
            ])
        else:
            # Shared模式：所有变量共享解码器
            self.decoder = CircularDecoder(self.hidden_channels, self.period_len)
    
    def _extract_features(self, x):
        """提取列特征"""
        # x: (batch, enc_in, num_periods, period_len)
        batch_size = x.size(0)
        
        if self.individual:
            # Individual模式
            features = []
            for var_idx in range(self.enc_in):
                var_features = []
                for period_idx in range(self.num_periods_input):
                    period_data = x[:, var_idx, period_idx, :]  # (batch, period_len)
                    feature = self.feature_extractors[var_idx](period_data)  # (batch, hidden_channels, period_len)
                    var_features.append(feature)
                var_features = torch.stack(var_features, dim=1)  # (batch, num_periods, hidden_channels, period_len)
                features.append(var_features)
            features = torch.stack(features, dim=1)  # (batch, enc_in, num_periods, hidden_channels, period_len)
        else:
            # Shared模式
            x_reshaped = x.reshape(batch_size * self.enc_in * self.num_periods_input, self.period_len)
            features_flat = self.feature_extractor(x_reshaped)  # (batch*enc_in*num_periods, hidden_channels, period_len)
            features = features_flat.reshape(batch_size, self.enc_in, self.num_periods_input, self.hidden_channels, self.period_len)
        
        return features
    
    def _apply_temporal_model(self, features):
        """应用时序模型"""
        # features: (batch, enc_in, num_periods_input, hidden_channels, period_len)
        batch_size = features.size(0)
        
        if self.individual:
            # Individual模式
            temporal_outputs = []
            for var_idx in range(self.enc_in):
                var_features = features[:, var_idx, :, :, :]  # (batch, num_periods_input, hidden_channels, period_len)
                
                if self.use_convlstm:
                    var_output, _ = self.temporal_models[var_idx](var_features)  # (batch, num_periods_input, hidden_channels, period_len)
                else:
                    var_output = self.temporal_models[var_idx](var_features)  # (batch, num_periods_input, hidden_channels, period_len)
                
                temporal_outputs.append(var_output)
            temporal_outputs = torch.stack(temporal_outputs, dim=1)  # (batch, enc_in, num_periods_input, hidden_channels, period_len)
        else:
            # Shared模式
            features_reshaped = features.reshape(batch_size * self.enc_in, self.num_periods_input, self.hidden_channels, self.period_len)
            
            if self.use_convlstm:
                outputs_flat, _ = self.temporal_model(features_reshaped)  # (batch*enc_in, num_periods_input, hidden_channels, period_len)
            else:
                outputs_flat = self.temporal_model(features_reshaped)  # (batch*enc_in, num_periods_input, hidden_channels, period_len)
            
            temporal_outputs = outputs_flat.reshape(batch_size, self.enc_in, self.num_periods_input, self.hidden_channels, self.period_len)
        
        return temporal_outputs
    
    def _generate_future_periods(self, temporal_features, num_steps):
        """生成未来周期"""
        # temporal_features: (batch, enc_in, num_periods_input, hidden_channels, period_len)
        batch_size = temporal_features.size(0)
        
        # 使用最后一个时间步的特征作为初始状态
        last_features = temporal_features[:, :, -1:, :, :]  # (batch, enc_in, 1, hidden_channels, period_len)
        
        # 生成未来步骤
        future_periods = []
        current_features = last_features
        
        for step in range(num_steps):
            if self.individual:
                # Individual模式
                step_outputs = []
                for var_idx in range(self.enc_in):
                    var_features = current_features[:, var_idx, :, :, :]  # (batch, 1, hidden_channels, period_len)
                    
                    # 通过时序模型
                    if self.use_convlstm:
                        var_output, _ = self.temporal_models[var_idx](var_features)  # (batch, 1, hidden_channels, period_len)
                    else:
                        var_output = self.temporal_models[var_idx](var_features)  # (batch, 1, hidden_channels, period_len)
                    
                    # 解码为时间序列
                    var_hidden = var_output[:, 0, :, :]  # (batch, hidden_channels, period_len)
                    var_decoded = self.decoders[var_idx](var_hidden)  # (batch, period_len)
                    step_outputs.append(var_decoded)
                
                step_output = torch.stack(step_outputs, dim=1)  # (batch, enc_in, period_len)
                
                # 更新current_features（使用预测结果）
                if step < num_steps - 1:  # 不是最后一步
                    # 重新编码预测结果作为下一步输入
                    next_features = []
                    for var_idx in range(self.enc_in):
                        var_pred = step_output[:, var_idx, :]  # (batch, period_len)
                        var_feature = self.feature_extractors[var_idx](var_pred)  # (batch, hidden_channels, period_len)
                        next_features.append(var_feature.unsqueeze(2))  # (batch, hidden_channels, 1, period_len)
                    current_features = torch.stack(next_features, dim=1)  # (batch, enc_in, 1, hidden_channels, period_len)
                    current_features = current_features.transpose(2, 3)  # (batch, enc_in, 1, hidden_channels, period_len)
            else:
                # Shared模式
                current_reshaped = current_features.reshape(batch_size * self.enc_in, 1, self.hidden_channels, self.period_len)
                
                # 通过时序模型
                if self.use_convlstm:
                    output_reshaped, _ = self.temporal_model(current_reshaped)  # (batch*enc_in, 1, hidden_channels, period_len)
                else:
                    output_reshaped = self.temporal_model(current_reshaped)  # (batch*enc_in, 1, hidden_channels, period_len)
                
                # 解码
                hidden_flat = output_reshaped[:, 0, :, :]  # (batch*enc_in, hidden_channels, period_len)
                decoded_flat = self.decoder(hidden_flat)  # (batch*enc_in, period_len)
                step_output = decoded_flat.reshape(batch_size, self.enc_in, self.period_len)
                
                # 更新current_features
                if step < num_steps - 1:
                    # 重新编码
                    next_features_flat = self.feature_extractor(decoded_flat)  # (batch*enc_in, hidden_channels, period_len)
                    current_features = next_features_flat.reshape(batch_size, self.enc_in, 1, self.hidden_channels, self.period_len)
            
            future_periods.append(step_output)
        
        # 堆叠所有未来周期
        future_periods = torch.stack(future_periods, dim=2)  # (batch, enc_in, num_steps, period_len)
        
        return future_periods
    
    def _compute_spectral_loss(self, pred, target):
        """计算频域损失"""
        # 计算FFT
        pred_fft = torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)
        
        # 幅度损失
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        mag_loss = F.l1_loss(pred_mag, target_mag)
        
        return mag_loss
    
    def _compute_smoothness_loss(self, pred):
        """计算列方向的平滑性损失（二阶导数惩罚）"""
        # pred: (batch, pred_len, enc_in)
        # 重塑为周期格式
        batch_size = pred.size(0)
        pred_padded = F.pad(pred.permute(0, 2, 1), (0, self.num_periods_output * self.period_len - self.pred_len), mode='replicate')
        pred_periods = pred_padded.reshape(batch_size, self.enc_in, self.num_periods_output, self.period_len)
        
        # 计算列方向的二阶导数
        smoothness_loss = 0.0
        for period_idx in range(self.num_periods_output):
            period_data = pred_periods[:, :, period_idx, :]  # (batch, enc_in, period_len)
            
            # 循环边界条件的二阶导数
            left = torch.roll(period_data, shifts=1, dims=-1)
            right = torch.roll(period_data, shifts=-1, dims=-1)
            second_derivative = left - 2 * period_data + right
            
            smoothness_loss += torch.mean(second_derivative ** 2)
        
        return smoothness_loss / self.num_periods_output
    
    def _update_scheduled_sampling_ratio(self):
        """更新Scheduled Sampling比例"""
        if not self.use_scheduled_sampling:
            return
        
        # 线性增长到最大值
        progress = min(1.0, self.training_step_count.float() / self.scheduled_sampling_decay_steps)
        self.scheduled_sampling_ratio = progress * self.max_scheduled_sampling_ratio
    
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
        
        # 提取列特征
        features = self._extract_features(x)  # (batch, enc_in, num_periods_input, hidden_channels, period_len)
        
        # 时序建模
        temporal_features = self._apply_temporal_model(features)  # (batch, enc_in, num_periods_input, hidden_channels, period_len)
        
        # 生成未来周期
        future_periods = self._generate_future_periods(temporal_features, self.num_periods_output)  # (batch, enc_in, num_periods_output, period_len)
        
        # 重塑回原始格式
        output = future_periods.reshape(batch_size, self.enc_in, -1)  # (batch, enc_in, total_pred_len)
        output = output[:, :, :self.pred_len]  # 截断到目标长度
        output = output.permute(0, 2, 1)  # (batch, pred_len, enc_in)
        
        return output
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        # 更新训练步数和Scheduled Sampling比例
        self.training_step_count += 1
        self._update_scheduled_sampling_ratio()
        
        dec_inp = self._build_decoder_input(batch_y)
        
        outputs = self(
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
        
        # 主要预测损失（使用Huber损失）
        main_loss = criterion(outputs, batch_y)
        
        # 频域损失
        spectral_loss = self._compute_spectral_loss(outputs, batch_y)
        
        # 平滑性损失
        smoothness_loss = self._compute_smoothness_loss(outputs)
        
        # 总损失
        total_loss = (main_loss + 
                     self.spectral_loss_weight * spectral_loss + 
                     self.smoothness_loss_weight * smoothness_loss)
        
        # 记录损失
        self.log("train_loss_main", main_loss, on_epoch=True)
        self.log("train_loss_spectral", spectral_loss, on_epoch=True)
        self.log("train_loss_smoothness", smoothness_loss, on_epoch=True)
        self.log("train_loss", total_loss, on_epoch=True)
        self.log("scheduled_sampling_ratio", self.scheduled_sampling_ratio, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)
        
        outputs = self(
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
        
        outputs = self(
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
