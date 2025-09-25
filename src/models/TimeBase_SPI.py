import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pl_bases.default_module import DefaultPLModule


def cal_orthogonal_loss(matrix):
    gram_matrix = torch.matmul(matrix.transpose(-2, -1), matrix)
    one_diag = torch.diagonal(gram_matrix, dim1=-2, dim2=-1)
    two_diag = torch.diag_embed(one_diag)
    off_diagonal = gram_matrix - two_diag
    loss = torch.norm(off_diagonal, dim=(-2, -1))
    return loss.mean()


class TimeBase_SPI(DefaultPLModule):
    """
    TimeBase Segment-Period with Intermediate supervision (SPI) 模型：
    带中间监督信号的逆序串行双维度降维时间序列预测模型
    
    策略特点：
    - 在SP基础上增加中间监督信号
    - 第一阶段（Segment降维）后添加中间预测头
    - 中间预测损失与最终预测损失联合训练
    - 引入门控机制调节两阶段信息流
    - 自适应权重平衡中间损失与最终损失
    
    创新设计：
    1. 中间监督信号：在Segment降维后产生中间预测
    2. 门控融合机制：控制第一阶段向第二阶段的信息传递
    3. 自适应损失权重：动态平衡两个监督信号的重要性
    4. 渐进式特征提取：先学习Segment模式，再优化Period模式
    
    适用场景：
    - 需要强化Segment特征学习的时间序列
    - 适合跨周期模式比period内模式更重要的数据
    - 希望通过中间监督提升全局特征学习
    """
    def __init__(self, configs):
        super(TimeBase_SPI, self).__init__(configs)
        
        # 基础参数
        self.use_period_norm = configs.use_period_norm
        self.use_segment_norm = getattr(configs, "use_segment_norm", False)
        self.use_orthogonal = configs.use_orthogonal
        
        # 正交权重配置（SP策略中segment权重更高）
        self.orthogonal_weight = getattr(configs, "orthogonal_weight", 0.04)
        self.segment_ortho_weight = getattr(configs, "segment_ortho_weight", self.orthogonal_weight)
        self.period_ortho_weight = getattr(configs, "period_ortho_weight", self.orthogonal_weight * 0.5)
        
        # 中间监督配置
        self.use_intermediate_supervision = getattr(configs, "use_intermediate_supervision", True)
        self.intermediate_loss_weight = getattr(configs, "intermediate_loss_weight", 0.3)
        self.use_adaptive_weights = getattr(configs, "use_adaptive_weights", True)
        self.use_gating = getattr(configs, "use_gating", True)
        
        # 模型参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.pad_seq_len = 0
        self.basis_num = configs.basis_num
        self.period_basis_num = getattr(configs, "period_basis_num", self.period_len)
        self.individual = configs.individual
        
        # SPI策略的归一化验证
        # SPI基于SP策略，需要支持精细的双阶段归一化：
        # 1. Segment reduction前：segment norm
        # 2. Segment reduction后：segment denorm + period norm  
        # 3. Period reduction后：period denorm
        # 建议配置：use_period_norm=True, use_segment_norm=True
        if not self.use_period_norm or not self.use_segment_norm:
            print(f"⚠️ SPI策略建议启用双归一化: use_period_norm=True, use_segment_norm=True")
            print(f"   当前配置: use_period_norm={self.use_period_norm}, use_segment_norm={self.use_segment_norm}")
        
        # 计算segment数量
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        if self.seq_len > self.seg_num_x * self.period_len:
            self.pad_seq_len = (self.seg_num_x + 1) * self.period_len - self.seq_len
            self.seg_num_x += 1
        if self.pred_len > self.seg_num_y * self.period_len:
            self.seg_num_y += 1
        
        # 设置线性层和新增组件
        self._setup_linear_layers()
        self._setup_intermediate_supervision_components()
        self._setup_gating_mechanism()
        self._setup_adaptive_weight_network(configs)

    def _setup_linear_layers(self):
        """设置Segment和Period降维的线性层，以及中间映射层"""
        if self.individual:
            # Individual模式：为每个变量创建独立的线性层
            # Segment降维层
            self.segment_ts2basis = nn.ModuleList()
            self.segment_basis2ts = nn.ModuleList()
            # Period降维层
            self.period_ts2basis = nn.ModuleList()
            self.period_basis2ts = nn.ModuleList()
            # 中间映射层：处理Segment降维后的维度变化
            self.segment_to_period_mapping = nn.ModuleList()
            
            for i in range(self.enc_in):
                # Segment降维（先执行）
                self.segment_ts2basis.append(
                    nn.Linear(self.seg_num_x, self.basis_num)
                )
                self.segment_basis2ts.append(
                    nn.Linear(self.basis_num, self.seg_num_y)
                )
                # Period降维（后执行）
                self.period_ts2basis.append(
                    nn.Linear(self.period_len, self.period_basis_num)
                )
                self.period_basis2ts.append(
                    nn.Linear(self.period_basis_num, self.period_len)
                )
                # 中间映射层：从seg_num_y映射回seg_num_x for period reduction
                self.segment_to_period_mapping.append(
                    nn.Linear(self.seg_num_y, self.seg_num_x)
                )
        else:
            # Shared模式：所有变量共享线性层
            # Segment降维层（先执行）
            self.segment_ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
            self.segment_basis2ts = nn.Linear(self.basis_num, self.seg_num_y)
            # Period降维层（后执行）
            self.period_ts2basis = nn.Linear(self.period_len, self.period_basis_num)
            self.period_basis2ts = nn.Linear(self.period_basis_num, self.period_len)
            # 中间映射层
            self.segment_to_period_mapping = nn.Linear(self.seg_num_y, self.seg_num_x)

    def _setup_intermediate_supervision_components(self):
        """设置中间监督组件 - 在Segment降维后进行预测"""
        if not self.use_intermediate_supervision:
            return
            
        # 中间预测头：将Segment降维后的特征直接映射到预测长度
        # 输入：(b, period_len, seg_num_y) -> 输出：(b, pred_len)
        if self.individual:
            self.intermediate_predictors = nn.ModuleList()
            for i in range(self.enc_in):
                # 压缩到一维时间序列
                self.intermediate_predictors.append(
                    nn.Sequential(
                        nn.Linear(self.period_len * self.seg_num_y, 128),  # 压缩特征
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, self.pred_len),  # 映射到预测长度
                    )
                )
        else:
            # Shared模式的中间预测头
            self.intermediate_predictor = nn.Sequential(
                nn.Linear(self.period_len * self.seg_num_y, 128),  # 压缩特征
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.pred_len),  # 映射到预测长度
            )

    def _setup_gating_mechanism(self):
        """设置门控机制"""
        if not self.use_gating:
            return
            
        # 门控网络：控制第一阶段向第二阶段的信息流
        gate_input_dim = self.period_len * self.seg_num_y * self.enc_in
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool1d(64),  # 自适应池化到固定大小
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def _setup_adaptive_weight_network(self, configs):
        """设置自适应权重网络"""
        if not self.use_adaptive_weights:
            return
            
        # 自适应权重网络：动态调整中间损失和最终损失的权重
        self.weight_network = nn.Sequential(
            nn.Linear(3, 16),  # [intermediate_error, final_error, training_progress]
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 训练进度计数器
        self.register_buffer('training_step_count', torch.tensor(0.0))
        self.max_training_steps = getattr(configs, "max_training_steps", 10000)

    def _apply_segment_reduction(self, x, b, c):
        """应用Segment维度降维 - 优化版本：批量处理减少循环"""
        if self.individual:
            x_reshaped = x.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_x)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_x)  # (c, b*period_len, seg_num_x)
            
            segment_basis_list = []
            x_output_list = []
            
            for i in range(self.enc_in):
                segment_basis = self.segment_ts2basis[i](x_flat[i])  # (b*period_len, basis_num)
                segment_pred = self.segment_basis2ts[i](segment_basis)  # (b*period_len, seg_num_y)
                
                segment_basis_list.append(segment_basis.reshape(b, self.period_len, self.basis_num))
                x_output_list.append(segment_pred.reshape(b, self.period_len, self.seg_num_y))
            
            x_output = torch.stack(x_output_list, dim=1)  # (b, c, period_len, seg_num_y)
            segment_basis_combined = torch.stack(segment_basis_list, dim=1)  # (b, c, period_len, basis_num)
            segment_basis_combined = segment_basis_combined.reshape(-1, self.period_len, self.basis_num)
            
        else:
            x_flat = x.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
            
            segment_basis = self.segment_ts2basis(x_flat)  # (b*c*period_len, basis_num)
            segment_pred = self.segment_basis2ts(segment_basis)  # (b*c*period_len, seg_num_y)
            
            x_output = segment_pred.reshape(b, c, self.period_len, self.seg_num_y)
            segment_basis_combined = segment_basis.reshape(-1, self.period_len, self.basis_num)
        
        return x_output, segment_basis_combined

    def _apply_period_reduction_with_mapping(self, x, b, c):
        """应用Period维度降维 - 带有中间映射层处理维度变化"""
        # Step 1: 先通过映射层将seg_num_y映射回seg_num_x进行period降维
        if self.individual:
            x_reshaped = x.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_y)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_y)  # (c, b*period_len, seg_num_y)
            
            x_mapped_list = []
            for i in range(self.enc_in):
                x_mapped = self.segment_to_period_mapping[i](x_flat[i])  # (b*period_len, seg_num_x)
                x_mapped_reshaped = x_mapped.reshape(b, self.period_len, self.seg_num_x)
                x_mapped_list.append(x_mapped_reshaped)
            
            x_mapped = torch.stack(x_mapped_list, dim=1)  # (b, c, period_len, seg_num_x)
        else:
            x_flat = x.reshape(-1, self.seg_num_y)  # (b*c*period_len, seg_num_y)
            x_mapped_flat = self.segment_to_period_mapping(x_flat)  # (b*c*period_len, seg_num_x)
            x_mapped = x_mapped_flat.reshape(b, c, self.period_len, self.seg_num_x)
        
        # Step 2: Period降维处理
        if self.individual:
            x_transposed = x_mapped.permute(0, 1, 3, 2)  # (b, c, seg_num_x, period_len)
            x_reshaped = x_transposed.permute(1, 0, 2, 3)  # (c, b, seg_num_x, period_len)
            x_flat = x_reshaped.reshape(c, -1, self.period_len)  # (c, b*seg_num_x, period_len)
            
            period_basis_list = []
            x_period_output_list = []
            
            for i in range(self.enc_in):
                period_coeffs = self.period_ts2basis[i](x_flat[i])  # (b*seg_num_x, period_basis_num)
                period_basis = period_coeffs.reshape(b, self.seg_num_x, self.period_basis_num)
                
                period_reconstructed = self.period_basis2ts[i](period_coeffs)  # (b*seg_num_x, period_len)
                period_pred_reshaped = period_reconstructed.reshape(b, self.seg_num_x, self.period_len)
                period_pred = period_pred_reshaped.permute(0, 2, 1)  # (b, period_len, seg_num_x)
                
                period_basis_list.append(period_basis)
                x_period_output_list.append(period_pred)
            
            x_period_output = torch.stack(x_period_output_list, dim=1)  # (b, c, period_len, seg_num_x)
            period_basis_combined = torch.stack(period_basis_list, dim=1)  # (b, c, seg_num_x, period_basis_num)
            period_basis_combined = period_basis_combined.reshape(-1, self.seg_num_x, self.period_basis_num)
            
        else:
            x_transposed = x_mapped.permute(0, 1, 3, 2)  # (b, c, seg_num_x, period_len)
            x_flat = x_transposed.reshape(-1, self.period_len)  # (b*c*seg_num_x, period_len)
            
            period_coeffs = self.period_ts2basis(x_flat)  # (b*c*seg_num_x, period_basis_num)
            period_basis = period_coeffs.reshape(b, c, self.seg_num_x, self.period_basis_num)
            
            period_reconstructed = self.period_basis2ts(period_coeffs)  # (b*c*seg_num_x, period_len)
            x_reconstructed = period_reconstructed.reshape(b, c, self.seg_num_x, self.period_len)
            x_period_output = x_reconstructed.permute(0, 1, 3, 2)  # (b, c, period_len, seg_num_x)
            
            period_basis_combined = period_basis.reshape(-1, self.seg_num_x, self.period_basis_num)
        
        # Step 3: 最终映射回seg_num_y
        if self.individual:
            x_reshaped = x_period_output.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_x)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_x)  # (c, b*period_len, seg_num_x)
            
            x_final_list = []
            for i in range(self.enc_in):
                x_final = self.segment_basis2ts[i](
                    self.segment_ts2basis[i](x_flat[i])
                )  # (b*period_len, seg_num_y)
                x_final_reshaped = x_final.reshape(b, self.period_len, self.seg_num_y)
                x_final_list.append(x_final_reshaped)
            
            x_output = torch.stack(x_final_list, dim=1)  # (b, c, period_len, seg_num_y)
        else:
            x_flat = x_period_output.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
            x_final_flat = self.segment_basis2ts(
                self.segment_ts2basis(x_flat)
            )  # (b*c*period_len, seg_num_y)
            x_output = x_final_flat.reshape(b, c, self.period_len, self.seg_num_y)
        
        return x_output, period_basis_combined

    def _generate_intermediate_prediction(self, x_segment_reduced, b, c):
        """生成中间预测 - 基于Segment降维的结果"""
        if not self.use_intermediate_supervision:
            return None
            
        # x_segment_reduced: (b, c, period_len, seg_num_y)
        if self.individual:
            intermediate_preds = []
            for i in range(self.enc_in):
                # 对每个变量单独处理
                var_data = x_segment_reduced[:, i, :, :]  # (b, period_len, seg_num_y)
                var_data_flat = var_data.reshape(b, -1)  # (b, period_len * seg_num_y)
                # 通过中间预测头
                intermediate_pred = self.intermediate_predictors[i](var_data_flat)  # (b, pred_len)
                intermediate_preds.append(intermediate_pred)
            
            # 合并所有变量的预测
            intermediate_output = torch.stack(intermediate_preds, dim=2)  # (b, pred_len, c)
        else:
            # Shared模式：所有变量使用相同的预测头
            x_flat = x_segment_reduced.reshape(b * c, -1)  # (b*c, period_len * seg_num_y)
            
            # 通过预测头
            intermediate_flat = self.intermediate_predictor(x_flat)  # (b*c, pred_len)
            intermediate_output = intermediate_flat.reshape(b, c, self.pred_len)
            intermediate_output = intermediate_output.permute(0, 2, 1)  # (b, pred_len, c)
        
        return intermediate_output

    def _apply_gating(self, x_segment_reduced, b, c):
        """应用门控机制"""
        if not self.use_gating:
            return x_segment_reduced
            
        # 计算门控权重
        x_flat = x_segment_reduced.reshape(b, -1)  # (b, c*period_len*seg_num_y)
        x_flat_unsqueezed = x_flat.unsqueeze(1)  # (b, 1, c*period_len*seg_num_y)
        
        # 通过自适应池化和门控网络
        pooled = self.gate_network[0](x_flat_unsqueezed).squeeze(1)  # (b, 64)
        gate_weights = self.gate_network[1:](pooled)  # (b, 1)
        
        # 应用门控 - 正确的维度扩展
        # x_segment_reduced: (b, c, period_len, seg_num_y)
        # gate_weights: (b, 1) -> (b, 1, 1, 1)
        gate_weights = gate_weights.view(b, 1, 1, 1)  # 确保正确的形状
        x_gated = x_segment_reduced * gate_weights
        
        return x_gated

    def _compute_adaptive_weight(self, intermediate_error, final_error):
        """计算自适应权重"""
        if not self.use_adaptive_weights:
            return self.intermediate_loss_weight
            
        # 训练进度
        training_progress = self.training_step_count / self.max_training_steps
        training_progress = torch.clamp(training_progress, 0.0, 1.0)
        
        # 构建输入特征
        features = torch.stack([
            intermediate_error.detach(),
            final_error.detach(), 
            training_progress
        ], dim=0).unsqueeze(0)  # (1, 3)
        
        # 计算自适应权重
        adaptive_weight = self.weight_network(features).squeeze()
        
        return adaptive_weight

    def _apply_segment_norm(self, x, b, c):
        """阶段1：Segment reduction前的segment归一化"""
        if self.use_segment_norm:
            segment_mean = torch.mean(x, dim=-1, keepdim=True)  # (bc, period_len, 1)
            x = x - segment_mean
            return x, {"segment_mean": segment_mean}
        else:
            x = x.reshape(b, c, -1)
            mean = torch.mean(x, dim=-1, keepdim=True)  # (b, c, 1)
            x = x - mean
            x = x.reshape(-1, self.period_len, self.seg_num_x)
            return x, {"mean": mean}

    def _apply_segment_denorm_and_period_norm(self, x, segment_norm_stats, b, c):
        """阶段2：Segment reduction后的segment逆归一化 + period归一化"""
        # Step 1: Segment逆归一化
        if self.use_segment_norm and "segment_mean" in segment_norm_stats:
            segment_mean = segment_norm_stats["segment_mean"]  # (bc, period_len, 1)
            x = x + segment_mean
        elif "mean" in segment_norm_stats:
            x = x.reshape(b, c, -1)
            x = x + segment_norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)

        # Step 2: Period归一化
        if self.use_period_norm:
            period_mean = torch.mean(x, dim=1, keepdim=True)  # (bc, 1, seg_num_y)
            x = x - period_mean
            return x, {"period_mean": period_mean}
        else:
            return x, {}

    def _apply_period_denorm(self, x, period_norm_stats, b, c):
        """阶段3：Period reduction后的period逆归一化"""
        if self.use_period_norm and "period_mean" in period_norm_stats:
            period_mean = period_norm_stats["period_mean"]  # (bc, 1, seg_num_y)
            x = x + period_mean
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """前向传播：带中间监督的逆序串行降维策略（先Segment，再Period）"""
        x = x_enc
        b, t, c = x.shape
        batch_size = b
        x = x.permute(0, 2, 1)  # b c t

        # Padding
        if self.pad_seq_len > 0:
            pad_start = (self.seg_num_x - 1) * self.period_len
            x = torch.cat(
                [x, x[:, :, pad_start - self.pad_seq_len : pad_start]], dim=-1
            )

        # Reshape
        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.period_len)
        x = x.permute(0, 1, 3, 2).reshape(
            -1, self.period_len, self.seg_num_x
        )  # (bc, p, n)

        # 阶段1: Segment reduction前的segment归一化
        x, segment_norm_stats = self._apply_segment_norm(x, b, c)
        
        # 阶段1: Segment降维
        x_4d = x.reshape(b, c, self.period_len, self.seg_num_x)
        x_segment_reduced, segment_basis = self._apply_segment_reduction(x_4d, b, c)
        
        # 🆕 中间监督：生成第一阶段的预测
        intermediate_prediction = self._generate_intermediate_prediction(x_segment_reduced, b, c)
        
        # 🆕 门控机制：控制信息流
        x_segment_gated = self._apply_gating(x_segment_reduced, b, c)
        
        # 阶段2: Segment reduction后的segment逆归一化 + period归一化
        x_flat = x_segment_gated.reshape(-1, self.period_len, self.seg_num_y)
        x_period_normed, period_norm_stats = self._apply_segment_denorm_and_period_norm(x_flat, segment_norm_stats, b, c)
        
        # 阶段2: Period降维
        x_4d_normed = x_period_normed.reshape(b, c, self.period_len, self.seg_num_y)
        x_period_reduced, period_basis = self._apply_period_reduction_with_mapping(x_4d_normed, b, c)

        # 阶段3: Period reduction后的period逆归一化
        x_final_flat = x_period_reduced.reshape(-1, self.period_len, self.seg_num_y)
        x_denormed = self._apply_period_denorm(x_final_flat, period_norm_stats, b, c)

        # 最终输出
        x = x_denormed.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y)
        x = x.permute(0, 1, 3, 2)  # (batch_size, enc_in, seg_num_y, period_len)
        x = x.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)  # b t c

        final_output = x[:, : self.pred_len, :]

        # 返回结果
        if self.use_intermediate_supervision:
            if self.use_orthogonal:
                return final_output, intermediate_prediction, segment_basis, period_basis
            else:
                return final_output, intermediate_prediction
        else:
            if self.use_orthogonal:
                return final_output, segment_basis, period_basis
            else:
                return final_output

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        # 更新训练步数计数器
        self.training_step_count += 1

        dec_inp = self._build_decoder_input(batch_y)

        forward_output = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        # 解析输出
        if self.use_intermediate_supervision:
            if self.use_orthogonal:
                final_outputs, intermediate_outputs, segment_basis, period_basis = forward_output
            else:
                final_outputs, intermediate_outputs = forward_output
                segment_basis = period_basis = None
        else:
            if self.use_orthogonal:
                final_outputs, segment_basis, period_basis = forward_output
            else:
                final_outputs = forward_output
                segment_basis = period_basis = None
            intermediate_outputs = None

        final_outputs = final_outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        
        # 计算最终损失
        final_loss = criterion(final_outputs, batch_y)
        
        # 计算总损失
        total_loss = final_loss
        
        # 🆕 中间监督损失
        if self.use_intermediate_supervision and intermediate_outputs is not None:
            intermediate_loss = criterion(intermediate_outputs, batch_y)
            
            # 🆕 自适应权重
            if self.use_adaptive_weights:
                intermediate_weight = self._compute_adaptive_weight(
                    intermediate_loss.mean(), final_loss.mean()
                )
            else:
                intermediate_weight = self.intermediate_loss_weight
            
            weighted_intermediate_loss = intermediate_weight * intermediate_loss
            total_loss = total_loss + weighted_intermediate_loss
            
            # 记录中间监督相关损失
            self.log("train_loss_intermediate", intermediate_loss, on_epoch=True)
            self.log("train_loss_intermediate_weighted", weighted_intermediate_loss, on_epoch=True)
            self.log("intermediate_weight", intermediate_weight, on_epoch=True)

        # 正交损失计算（SP策略中segment权重更高）
        if self.use_orthogonal:
            total_orthogonal_loss = 0
            
            if segment_basis is not None:
                segment_orthogonal_loss = cal_orthogonal_loss(segment_basis)
                weighted_segment_loss = self.segment_ortho_weight * segment_orthogonal_loss
                total_orthogonal_loss += weighted_segment_loss
                self.log("train_loss_segment_orthogonal", segment_orthogonal_loss, on_epoch=True)
                self.log("train_loss_segment_weighted", weighted_segment_loss, on_epoch=True)
            
            if period_basis is not None:
                period_orthogonal_loss = cal_orthogonal_loss(period_basis)
                weighted_period_loss = self.period_ortho_weight * period_orthogonal_loss
                total_orthogonal_loss += weighted_period_loss
                self.log("train_loss_period_orthogonal", period_orthogonal_loss, on_epoch=True)
                self.log("train_loss_period_weighted", weighted_period_loss, on_epoch=True)
            
            total_loss = total_loss + total_orthogonal_loss
            self.log("train_loss_orthogonal_total", total_orthogonal_loss, on_epoch=True)

        # 记录主要损失
        self.log("train_loss_final", final_loss, on_epoch=True)
        self.log("train_loss", total_loss, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)

        forward_output = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        # 验证时只使用最终输出
        if self.use_intermediate_supervision:
            if self.use_orthogonal:
                outputs, _, _, _ = forward_output
            else:
                outputs, _ = forward_output
        else:
            if self.use_orthogonal:
                outputs, _, _ = forward_output
            else:
                outputs = forward_output

        outputs = outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

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

        forward_output = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        # 测试时只使用最终输出
        if self.use_intermediate_supervision:
            if self.use_orthogonal:
                outputs, _, _, _ = forward_output
            else:
                outputs, _ = forward_output
        else:
            if self.use_orthogonal:
                outputs, _, _ = forward_output
            else:
                outputs = forward_output

        outputs = outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        pred = outputs.detach()
        true = batch_y.detach()

        from src.utils.metrics import metric

        loss = metric(pred, true)
        self.log_dict({f"test_{k}": v for k, v in loss.items()}, on_epoch=True)

        return loss
