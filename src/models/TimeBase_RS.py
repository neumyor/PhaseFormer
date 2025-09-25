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


class TimeBase_RS(DefaultPLModule):
    """
    TimeSeries Residual Split (RS) 模型：基于残差分解的时间序列预测模型
    
    策略特点：
    - 输入序列首先通过period_len长度的滑动窗口进行平滑化
    - 平滑后的数据（趋势成分）经过Segment reduction
    - 残差（高频成分）经过Period reduction  
    - 两个分支各自独立归一化处理
    - 最终融合两个分支的输出
    
    设计思想：
    - 平滑数据包含主要趋势，适合Segment reduction捕获跨segment模式
    - 残差包含高频细节，适合Period reduction捕获period内模式
    - 分离处理不同频率成分，提高建模精度
    
    适用场景：
    - 包含明显趋势和高频噪声的时间序列
    - 需要分别建模不同频率成分的数据
    - 对细节和趋势都有较高要求的预测任务
    """
    def __init__(self, configs):
        super(TimeBase_RS, self).__init__(configs)
        
        # 基础参数
        self.use_period_norm = getattr(configs, "use_period_norm", False)  # 残差分支使用
        self.use_segment_norm = getattr(configs, "use_segment_norm", False)  # 平滑分支使用
        self.use_orthogonal = configs.use_orthogonal
        
        # 正交权重配置
        self.orthogonal_weight = getattr(configs, "orthogonal_weight", 0.04)
        self.segment_ortho_weight = getattr(configs, "segment_ortho_weight", self.orthogonal_weight)
        self.period_ortho_weight = getattr(configs, "period_ortho_weight", self.orthogonal_weight * 0.5)
        
        # 模型参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.pad_seq_len = 0
        self.basis_num = configs.basis_num
        self.period_basis_num = getattr(configs, "period_basis_num", self.period_len)
        self.individual = configs.individual
        
        # 计算segment数量
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        if self.seq_len > self.seg_num_x * self.period_len:
            self.pad_seq_len = (self.seg_num_x + 1) * self.period_len - self.seq_len
            self.seg_num_x += 1
        if self.pred_len > self.seg_num_y * self.period_len:
            self.seg_num_y += 1
        
        # 平滑窗口（period_len长度的均值滤波）
        self.smooth_kernel_size = self.period_len
        
        # 设置线性层
        self._setup_linear_layers()

    def _setup_linear_layers(self):
        """设置平滑分支（Segment）和残差分支（Period）的线性层"""
        if self.individual:
            # Individual模式：为每个变量创建独立的线性层
            # 平滑分支（Segment reduction）
            self.smooth_ts2basis = nn.ModuleList()
            self.smooth_basis2ts = nn.ModuleList()
            # 残差分支（Period reduction）
            self.residual_ts2basis = nn.ModuleList()
            self.residual_basis2ts = nn.ModuleList()
            self.residual_seg_mapping = nn.ModuleList()
            
            for i in range(self.enc_in):
                # 平滑分支（Segment降维）
                self.smooth_ts2basis.append(
                    nn.Linear(self.seg_num_x, self.basis_num)
                )
                self.smooth_basis2ts.append(
                    nn.Linear(self.basis_num, self.seg_num_y)
                )
                # 残差分支（Period降维）
                self.residual_ts2basis.append(
                    nn.Linear(self.period_len, self.period_basis_num)
                )
                self.residual_basis2ts.append(
                    nn.Linear(self.period_basis_num, self.period_len)
                )
                # 残差分支输出映射：seg_num_x -> seg_num_y
                self.residual_seg_mapping.append(
                    nn.Linear(self.seg_num_x, self.seg_num_y)
                )
        else:
            # Shared模式：所有变量共享线性层
            # 平滑分支（Segment降维）
            self.smooth_ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
            self.smooth_basis2ts = nn.Linear(self.basis_num, self.seg_num_y)
            # 残差分支（Period降维）
            self.residual_ts2basis = nn.Linear(self.period_len, self.period_basis_num)
            self.residual_basis2ts = nn.Linear(self.period_basis_num, self.period_len)
            self.residual_seg_mapping = nn.Linear(self.seg_num_x, self.seg_num_y)

    def _apply_smoothing(self, x):
        """
        应用滑动窗口平滑化
        
        输入: x (batch_size, enc_in, seq_len)
        输出: smooth_data, residual_data
        """
        batch_size, enc_in, seq_len = x.shape
        window_size = self.smooth_kernel_size
        
        # 使用unfold进行滑动窗口操作
        # unfold: (batch_size, enc_in, seq_len) -> (batch_size, enc_in, seq_len-window_size+1, window_size)
        half_window = window_size // 2
        
        # 为了保持原始长度，在两端进行padding
        x_padded = F.pad(x, (half_window, half_window), mode='reflect')
        
        # 使用unfold获取滑动窗口
        windows = x_padded.unfold(dimension=2, size=window_size, step=1)
        
        # 计算每个窗口的均值
        smooth_data = windows.mean(dim=-1)
        
        # 确保输出长度与输入相同
        if smooth_data.size(-1) != seq_len:
            smooth_data = smooth_data[:, :, :seq_len]
        
        # 计算残差
        residual_data = x - smooth_data
        
        return smooth_data, residual_data

    def _apply_smooth_branch_segment_reduction(self, smooth_data, b, c):
        """
        平滑分支：应用Segment维度降维 - 优化版本：批量处理减少循环
        
        输入: smooth_data (b, c, period_len, seg_num_x)
        输出: x_output (b, c, period_len, seg_num_y), segment_basis_combined
        """
        if self.individual:
            # Individual模式：批量处理所有变量
            # smooth_data shape: (b, c, period_len, seg_num_x)
            x_reshaped = smooth_data.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_x)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_x)  # (c, b*period_len, seg_num_x)
            
            # 预分配输出张量
            segment_basis_list = []
            x_output_list = []
            
            for i in range(self.enc_in):
                segment_basis = self.smooth_ts2basis[i](x_flat[i])  # (b*period_len, basis_num)
                segment_pred = self.smooth_basis2ts[i](segment_basis)  # (b*period_len, seg_num_y)
                
                segment_basis_list.append(segment_basis.reshape(b, self.period_len, self.basis_num))
                x_output_list.append(segment_pred.reshape(b, self.period_len, self.seg_num_y))
            
            # 批量堆叠结果
            x_output = torch.stack(x_output_list, dim=1)  # (b, c, period_len, seg_num_y)
            segment_basis_combined = torch.stack(segment_basis_list, dim=1)  # (b, c, period_len, basis_num)
            segment_basis_combined = segment_basis_combined.reshape(-1, self.period_len, self.basis_num)
            
        else:
            # Shared模式：完全向量化处理
            # smooth_data shape: (b, c, period_len, seg_num_x)
            x_flat = smooth_data.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
            
            # 批量处理所有数据
            segment_basis = self.smooth_ts2basis(x_flat)  # (b*c*period_len, basis_num)
            segment_pred = self.smooth_basis2ts(segment_basis)  # (b*c*period_len, seg_num_y)
            
            # 重新整形输出
            x_output = segment_pred.reshape(b, c, self.period_len, self.seg_num_y)
            segment_basis_combined = segment_basis.reshape(-1, self.period_len, self.basis_num)
        
        return x_output, segment_basis_combined

    def _apply_residual_branch_period_reduction(self, residual_data, b, c):
        """
        残差分支：应用Period维度降维 - 优化版本：批量处理减少循环
        
        输入: residual_data (b, c, period_len, seg_num_x)
        输出: x_output (b, c, period_len, seg_num_y), period_basis_combined
        """
        if self.individual:
            # Individual模式：批量处理所有变量的Period降维
            # residual_data shape: (b, c, period_len, seg_num_x)
            x_transposed = residual_data.permute(0, 1, 3, 2)  # (b, c, seg_num_x, period_len)
            x_reshaped = x_transposed.permute(1, 0, 2, 3)  # (c, b, seg_num_x, period_len)
            x_flat = x_reshaped.reshape(c, -1, self.period_len)  # (c, b*seg_num_x, period_len)
            
            # 批量处理所有变量
            period_basis_list = []
            x_period_output_list = []
            
            for i in range(self.enc_in):
                # 获取系数并重建
                period_coeffs = self.residual_ts2basis[i](x_flat[i])  # (b*seg_num_x, period_basis_num)
                period_basis = period_coeffs.reshape(b, self.seg_num_x, self.period_basis_num)
                
                period_reconstructed = self.residual_basis2ts[i](period_coeffs)  # (b*seg_num_x, period_len)
                period_pred_reshaped = period_reconstructed.reshape(b, self.seg_num_x, self.period_len)
                period_pred = period_pred_reshaped.permute(0, 2, 1)  # (b, period_len, seg_num_x)
                
                period_basis_list.append(period_basis)
                x_period_output_list.append(period_pred)
            
            # 批量堆叠结果
            x_period_output = torch.stack(x_period_output_list, dim=1)  # (b, c, period_len, seg_num_x)
            period_basis_combined = torch.stack(period_basis_list, dim=1)  # (b, c, seg_num_x, period_basis_num)
            period_basis_combined = period_basis_combined.reshape(-1, self.seg_num_x, self.period_basis_num)
            
        else:
            # Shared模式：完全向量化处理
            x_transposed = residual_data.permute(0, 1, 3, 2)  # (b, c, seg_num_x, period_len)
            x_flat = x_transposed.reshape(-1, self.period_len)  # (b*c*seg_num_x, period_len)
            
            period_coeffs = self.residual_ts2basis(x_flat)  # (b*c*seg_num_x, period_basis_num)
            period_basis = period_coeffs.reshape(b, c, self.seg_num_x, self.period_basis_num)
            
            period_reconstructed = self.residual_basis2ts(period_coeffs)  # (b*c*seg_num_x, period_len)
            x_reconstructed = period_reconstructed.reshape(b, c, self.seg_num_x, self.period_len)
            x_period_output = x_reconstructed.permute(0, 1, 3, 2)  # (b, c, period_len, seg_num_x)
            
            period_basis_combined = period_basis.reshape(-1, self.seg_num_x, self.period_basis_num)
        
        # 向量化映射到目标segment数量 (seg_num_x -> seg_num_y)
        if self.individual:
            # Individual模式：批量处理所有变量的映射
            x_reshaped = x_period_output.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_x)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_x)  # (c, b*period_len, seg_num_x)
            
            x_output_list = []
            for i in range(self.enc_in):
                period_mapped = self.residual_seg_mapping[i](x_flat[i])  # (b*period_len, seg_num_y)
                period_reshaped = period_mapped.reshape(b, self.period_len, self.seg_num_y)
                x_output_list.append(period_reshaped)
            
            x_output = torch.stack(x_output_list, dim=1)  # (b, c, period_len, seg_num_y)
        else:
            # Shared模式：完全向量化映射
            x_flat = x_period_output.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
            period_mapped = self.residual_seg_mapping(x_flat)  # (b*c*period_len, seg_num_y)
            x_output = period_mapped.reshape(b, c, self.period_len, self.seg_num_y)
        
        return x_output, period_basis_combined

    def _normalize_smooth_branch(self, x, b, c):
        """平滑分支的归一化（支持segment_norm和全局归一化）"""
        if self.use_segment_norm:
            # Segment归一化：在seg_num维度求均值
            # x: (bc, period_len, seg_num_x)，在seg_num维度求均值
            segment_mean = torch.mean(x, dim=-1, keepdim=True)  # (bc, period_len, 1)
            x = x - segment_mean
            return x, {"segment_mean": segment_mean}
        else:
            # 全局归一化
            x = x.reshape(b, c, -1)
            mean = torch.mean(x, dim=-1, keepdim=True)  # (b, c, 1)
            x = x - mean
            x = x.reshape(-1, self.period_len, self.seg_num_x)
            return x, {"mean": mean}

    def _normalize_residual_branch(self, x, b, c):
        """残差分支的归一化（支持period_norm和全局归一化）"""
        if self.use_period_norm:
            # Period归一化：在period_len维度求均值
            # x: (bc, period_len, seg_num_x)，在period_len维度求均值
            period_mean = torch.mean(x, dim=1, keepdim=True)  # (bc, 1, seg_num_x)
            x = x - period_mean
            return x, {"period_mean": period_mean}
        else:
            # 全局归一化
            x = x.reshape(b, c, -1)
            mean = torch.mean(x, dim=-1, keepdim=True)  # (b, c, 1)
            x = x - mean
            x = x.reshape(-1, self.period_len, self.seg_num_x)
            return x, {"mean": mean}

    def _denormalize_smooth_branch(self, x, norm_stats, b, c):
        """平滑分支的逆归一化"""
        if "segment_mean" in norm_stats:
            # Segment归一化的逆操作
            # x的形状：(bc, period_len, seg_num_y)
            # segment_mean的形状：(bc, period_len, 1)
            segment_mean = norm_stats["segment_mean"]  # (bc, period_len, 1)
            x = x + segment_mean
        else:
            # 全局归一化的逆操作
            x = x.reshape(b, c, -1)
            x = x + norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        return x

    def _denormalize_residual_branch(self, x, norm_stats, b, c):
        """残差分支的逆归一化"""
        if "period_mean" in norm_stats:
            # Period归一化的逆操作
            # x的形状：(bc, period_len, seg_num_y)
            # period_mean的形状：(bc, 1, seg_num_x)
            period_mean = norm_stats["period_mean"]  # (bc, 1, seg_num_x)
            
            if self.seg_num_x == self.seg_num_y:
                # segment数量相同，直接加回
                x = x + period_mean
            else:
                # segment数量不同，需要插值period_mean
                period_mean_reshaped = period_mean.reshape(b, c, 1, self.seg_num_x)
                period_mean_expanded = F.interpolate(
                    period_mean_reshaped.squeeze(2),  # (b, c, seg_num_x)
                    size=self.seg_num_y,
                    mode='linear',
                    align_corners=False
                ).unsqueeze(2)  # (b, c, 1, seg_num_y)
                period_mean_expanded = period_mean_expanded.reshape(-1, 1, self.seg_num_y)
                x = x + period_mean_expanded
        else:
            # 全局归一化的逆操作
            x = x.reshape(b, c, -1)
            x = x + norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        前向传播：残差分解策略
        """
        x = x_enc
        b, t, c = x.shape
        batch_size = b
        x = x.permute(0, 2, 1)  # b c t

        # Step 1: 滑动窗口平滑化，分解为平滑和残差
        smooth_data, residual_data = self._apply_smoothing(x)  # (b, c, t)

        # Padding（对两个分支都进行相同的padding）
        if self.pad_seq_len > 0:
            pad_start = (self.seg_num_x - 1) * self.period_len
            smooth_data = torch.cat(
                [smooth_data, smooth_data[:, :, pad_start - self.pad_seq_len : pad_start]], dim=-1
            )
            residual_data = torch.cat(
                [residual_data, residual_data[:, :, pad_start - self.pad_seq_len : pad_start]], dim=-1
            )

        # Reshape两个分支的数据
        smooth_data = smooth_data.reshape(batch_size, self.enc_in, self.seg_num_x, self.period_len)
        smooth_data = smooth_data.permute(0, 1, 3, 2).reshape(-1, self.period_len, self.seg_num_x)  # (bc, p, n)
        
        residual_data = residual_data.reshape(batch_size, self.enc_in, self.seg_num_x, self.period_len)
        residual_data = residual_data.permute(0, 1, 3, 2).reshape(-1, self.period_len, self.seg_num_x)  # (bc, p, n)

        # 残差分解双分支处理策略 - 内存布局优化版本:
        # Step 2: 分支归一化
        smooth_data, smooth_norm_stats = self._normalize_smooth_branch(smooth_data, b, c)
        residual_data, residual_norm_stats = self._normalize_residual_branch(residual_data, b, c)

        # Step 3: 分支降维处理 - 保持高效的张量布局
        smooth_data_4d = smooth_data.reshape(b, c, self.period_len, self.seg_num_x)
        residual_data_4d = residual_data.reshape(b, c, self.period_len, self.seg_num_x)
        
        # 平滑分支：Segment降维（批量处理优化）
        smooth_reduced, segment_basis = self._apply_smooth_branch_segment_reduction(smooth_data_4d, b, c)
        
        # 残差分支：Period降维（批量处理优化）
        residual_reduced, period_basis = self._apply_residual_branch_period_reduction(residual_data_4d, b, c)

        # Step 4: 分支逆归一化 - 直接在张量上操作
        smooth_output_flat = smooth_reduced.reshape(-1, self.period_len, self.seg_num_y)
        smooth_denormed = self._denormalize_smooth_branch(smooth_output_flat, smooth_norm_stats, b, c)
        
        residual_output_flat = residual_reduced.reshape(-1, self.period_len, self.seg_num_y)
        residual_denormed = self._denormalize_residual_branch(residual_output_flat, residual_norm_stats, b, c)

        # Step 5: 融合两个分支
        final_output = smooth_denormed + residual_denormed

        # Reshape back
        final_output = final_output.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y)
        final_output = final_output.permute(0, 1, 3, 2)  # (batch_size, enc_in, seg_num_y, period_len)
        final_output = final_output.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)  # b t c

        # 输出处理
        output = final_output[:, : self.pred_len, :]

        # 如果需要正交损失，返回两个basis
        if self.use_orthogonal:
            return output, segment_basis, period_basis
        else:
            return output

    def training_step(self, batch, batch_idx):
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

        if self.use_orthogonal:
            outputs, segment_basis, period_basis = forward_output
        else:
            outputs = forward_output

        outputs = outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        loss = criterion(outputs, batch_y)

        # 正交损失计算（分层权重）
        if self.use_orthogonal:
            total_orthogonal_loss = 0
            
            # Segment basis的正交损失（平滑分支）
            if segment_basis is not None:
                segment_orthogonal_loss = cal_orthogonal_loss(segment_basis)
                weighted_segment_loss = self.segment_ortho_weight * segment_orthogonal_loss
                total_orthogonal_loss += weighted_segment_loss
                self.log("train_loss_smooth_segment_orthogonal", segment_orthogonal_loss, on_epoch=True)
                self.log("train_loss_smooth_segment_weighted", weighted_segment_loss, on_epoch=True)
            
            # Period basis的正交损失（残差分支）
            if period_basis is not None:
                period_orthogonal_loss = cal_orthogonal_loss(period_basis)
                weighted_period_loss = self.period_ortho_weight * period_orthogonal_loss
                total_orthogonal_loss += weighted_period_loss
                self.log("train_loss_residual_period_orthogonal", period_orthogonal_loss, on_epoch=True)
                self.log("train_loss_residual_period_weighted", weighted_period_loss, on_epoch=True)
            
            total_loss = loss + total_orthogonal_loss
            
            self.log("train_loss_main", loss, on_epoch=True)
            self.log("train_loss_orthogonal_total", total_orthogonal_loss, on_epoch=True)
            self.log("train_loss", total_loss, on_epoch=True)
            
            return total_loss
        else:
            self.log("train_loss", loss, on_epoch=True)
            return loss

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
