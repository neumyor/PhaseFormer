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


class TimeBase_PSPara(DefaultPLModule):
    """
    TimeBase Period-Segment Parallel (PSPara) 模型：并行双维度降维的时间序列预测模型
    
    策略特点：
    - 使用Period和Segment双维度降维，并行执行
    - Period和Segment降维并行处理，结果融合
    - 每个分支独立归一化和逆归一化
    - 计算复杂度最高，内存使用最高
    - 支持分支特定的归一化策略
    
    归一化支持：
    - Period分支：支持period_norm和全局归一化（use_period_norm控制）
    - Segment分支：支持segment_norm和全局归一化（use_segment_norm控制）
    - 两个分支可以使用不同的归一化策略（通过通用参数控制）
    
    适用场景：
    - Period和Segment独立建模，追求最佳性能
    - 两个维度的特征相对独立
    - 有充足的计算资源
    - 需要对不同分支使用不同归一化策略的场景
    """
    def __init__(self, configs):
        super(TimeBase_PSPara, self).__init__(configs)
        
        # 基础参数
        self.use_period_norm = configs.use_period_norm
        self.use_segment_norm = getattr(configs, "use_segment_norm", False)
        self.use_orthogonal = configs.use_orthogonal
        
        # 正交权重配置
        self.orthogonal_weight = getattr(configs, "orthogonal_weight", 0.04)
        self.segment_ortho_weight = getattr(configs, "segment_ortho_weight", self.orthogonal_weight)
        self.period_ortho_weight = getattr(configs, "period_ortho_weight", self.orthogonal_weight * 0.3)
        
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
        
        # 设置线性层
        self._setup_linear_layers()

    def _setup_linear_layers(self):
        """设置Period和Segment降维的线性层，以及映射层"""
        if self.individual:
            # Individual模式：为每个变量创建独立的线性层
            # Period降维层
            self.period_ts2basis = nn.ModuleList()
            self.period_basis2ts = nn.ModuleList()
            self.period_seg_mapping = nn.ModuleList()
            # Segment降维层
            self.segment_ts2basis = nn.ModuleList()
            self.segment_basis2ts = nn.ModuleList()
            
            for i in range(self.enc_in):
                # Period降维
                self.period_ts2basis.append(
                    nn.Linear(self.period_len, self.period_basis_num)
                )
                self.period_basis2ts.append(
                    nn.Linear(self.period_basis_num, self.period_len)
                )
                # Period映射层：seg_num_x -> seg_num_y
                self.period_seg_mapping.append(
                    nn.Linear(self.seg_num_x, self.seg_num_y)
                )
                # Segment降维
                self.segment_ts2basis.append(
                    nn.Linear(self.seg_num_x, self.basis_num)
                )
                self.segment_basis2ts.append(
                    nn.Linear(self.basis_num, self.seg_num_y)
                )
        else:
            # Shared模式：所有变量共享线性层
            # Period降维层
            self.period_ts2basis = nn.Linear(self.period_len, self.period_basis_num)
            self.period_basis2ts = nn.Linear(self.period_basis_num, self.period_len)
            self.period_seg_mapping = nn.Linear(self.seg_num_x, self.seg_num_y)
            # Segment降维层
            self.segment_ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
            self.segment_basis2ts = nn.Linear(self.basis_num, self.seg_num_y)

    def _apply_period_reduction(self, x, b, c):
        """应用Period维度降维 - 优化版本：批量处理减少循环"""
        if self.individual:
            # Individual模式：批量处理所有变量的Period降维
            # x shape: (b, c, period_len, seg_num_x)
            x_transposed = x.permute(0, 1, 3, 2)  # (b, c, seg_num_x, period_len)
            x_reshaped = x_transposed.permute(1, 0, 2, 3)  # (c, b, seg_num_x, period_len)
            x_flat = x_reshaped.reshape(c, -1, self.period_len)  # (c, b*seg_num_x, period_len)
            
            # 批量处理所有变量
            period_basis_list = []
            x_output_list = []
            
            for i in range(self.enc_in):
                # 获取系数并重建
                period_coeffs = self.period_ts2basis[i](x_flat[i])  # (b*seg_num_x, period_basis_num)
                period_basis = period_coeffs.reshape(b, self.seg_num_x, self.period_basis_num)
                
                period_reconstructed = self.period_basis2ts[i](period_coeffs)  # (b*seg_num_x, period_len)
                period_pred_reshaped = period_reconstructed.reshape(b, self.seg_num_x, self.period_len)
                period_pred = period_pred_reshaped.permute(0, 2, 1)  # (b, period_len, seg_num_x)
                
                period_basis_list.append(period_basis)
                x_output_list.append(period_pred)
            
            # 批量堆叠结果
            x_output = torch.stack(x_output_list, dim=1)  # (b, c, period_len, seg_num_x)
            period_basis_combined = torch.stack(period_basis_list, dim=1)  # (b, c, seg_num_x, period_basis_num)
            period_basis_combined = period_basis_combined.reshape(-1, self.seg_num_x, self.period_basis_num)
            
        else:
            # Shared模式：完全向量化处理
            x_transposed = x.permute(0, 1, 3, 2)  # (b, c, seg_num_x, period_len)
            x_flat = x_transposed.reshape(-1, self.period_len)  # (b*c*seg_num_x, period_len)
            
            period_coeffs = self.period_ts2basis(x_flat)  # (b*c*seg_num_x, period_basis_num)
            period_basis = period_coeffs.reshape(b, c, self.seg_num_x, self.period_basis_num)
            
            period_reconstructed = self.period_basis2ts(period_coeffs)  # (b*c*seg_num_x, period_len)
            x_reconstructed = period_reconstructed.reshape(b, c, self.seg_num_x, self.period_len)
            x_output = x_reconstructed.permute(0, 1, 3, 2)  # (b, c, period_len, seg_num_x)
            
            period_basis_combined = period_basis.reshape(-1, self.seg_num_x, self.period_basis_num)
        
        return x_output, period_basis_combined

    def _apply_segment_reduction(self, x, b, c):
        """应用Segment维度降维 - 优化版本：批量处理减少循环"""
        if self.individual:
            # Individual模式：批量处理所有变量
            # x shape: (b, c, period_len, seg_num_x)
            x_reshaped = x.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_x)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_x)  # (c, b*period_len, seg_num_x)
            
            # 预分配输出张量
            segment_basis_list = []
            x_output_list = []
            
            for i in range(self.enc_in):
                segment_basis = self.segment_ts2basis[i](x_flat[i])  # (b*period_len, basis_num)
                segment_pred = self.segment_basis2ts[i](segment_basis)  # (b*period_len, seg_num_y)
                
                segment_basis_list.append(segment_basis.reshape(b, self.period_len, self.basis_num))
                x_output_list.append(segment_pred.reshape(b, self.period_len, self.seg_num_y))
            
            # 批量堆叠结果
            x_output = torch.stack(x_output_list, dim=1)  # (b, c, period_len, seg_num_y)
            segment_basis_combined = torch.stack(segment_basis_list, dim=1)  # (b, c, period_len, basis_num)
            segment_basis_combined = segment_basis_combined.reshape(-1, self.period_len, self.basis_num)
            
        else:
            # Shared模式：完全向量化处理
            # x shape: (b, c, period_len, seg_num_x)
            x_flat = x.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
            
            # 批量处理所有数据
            segment_basis = self.segment_ts2basis(x_flat)  # (b*c*period_len, basis_num)
            segment_pred = self.segment_basis2ts(segment_basis)  # (b*c*period_len, seg_num_y)
            
            # 重新整形输出
            x_output = segment_pred.reshape(b, c, self.period_len, self.seg_num_y)
            segment_basis_combined = segment_basis.reshape(-1, self.period_len, self.basis_num)
        
        return x_output, segment_basis_combined

    def _normalize_for_period_branch(self, x, b, c):
        """Period分支的独立归一化（支持period_norm和全局归一化）"""
        if self.use_period_norm:
            # Period归一化：在period_len维度求均值
            # x的形状：(bc, period_len, seg_num_x)
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

    def _normalize_for_segment_branch(self, x, b, c):
        """Segment分支的独立归一化"""
        if self.use_segment_norm:
            # Segment归一化：在seg_num维度求均值
            # x的形状：(bc, period_len, seg_num_x)
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

    def _denormalize_for_period_branch(self, x, norm_stats, b, c):
        """Period分支的独立逆归一化"""
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
            return x
        else:
            # 全局归一化的逆操作
            x = x.reshape(b, c, -1)
            x = x + norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
            return x

    def _denormalize_for_segment_branch(self, x, norm_stats, b, c):
        """Segment分支的独立逆归一化"""
        if "segment_mean" in norm_stats:
            # Segment归一化的逆操作
            # x的形状：(bc, period_len, seg_num_y)
            # segment_mean的形状：(bc, period_len, 1)
            segment_mean = norm_stats["segment_mean"]  # (bc, period_len, 1)
            x = x + segment_mean
            return x
        else:
            # 全局归一化的逆操作
            x = x.reshape(b, c, -1)
            x = x + norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
            return x

    def _apply_parallel_reduction(self, x, b, c):
        """
        应用并行降维策略 - Period和Segment降维并行执行，每个分支独立归一化
        """
        # 为两个分支创建独立的数据副本
        x_period_branch = x.clone()
        x_segment_branch = x.clone()
        
        # Period分支处理
        # Step 1a: Period分支独立归一化
        x_period_flat = x_period_branch.reshape(-1, self.period_len, self.seg_num_x)
        x_period_norm, period_norm_stats = self._normalize_for_period_branch(x_period_flat, b, c)
        x_period_branch = x_period_norm.reshape(b, c, self.period_len, self.seg_num_x)
        
        # Step 1b: Period降维
        x_period, period_basis_combined = self._apply_period_reduction(x_period_branch, b, c)
        
        # Step 1c: Period结果映射到seg_num_y - 向量化优化
        if self.individual:
            # Individual模式：批量处理所有变量的映射
            x_reshaped = x_period.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_x)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_x)  # (c, b*period_len, seg_num_x)
            
            x_period_mapped_list = []
            for i in range(self.enc_in):
                period_mapped = self.period_seg_mapping[i](x_flat[i])  # (b*period_len, seg_num_y)
                period_reshaped = period_mapped.reshape(b, self.period_len, self.seg_num_y)
                x_period_mapped_list.append(period_reshaped)
            
            x_period_mapped = torch.stack(x_period_mapped_list, dim=1)  # (b, c, period_len, seg_num_y)
        else:
            # Shared模式：完全向量化映射
            x_flat = x_period.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
            period_mapped = self.period_seg_mapping(x_flat)  # (b*c*period_len, seg_num_y)
            x_period_mapped = period_mapped.reshape(b, c, self.period_len, self.seg_num_y)
        
        # Step 1d: Period分支逆归一化
        x_period_final_flat = x_period_mapped.reshape(-1, self.period_len, self.seg_num_y)
        x_period_final_flat = self._denormalize_for_period_branch(x_period_final_flat, period_norm_stats, b, c)
        x_period_final = x_period_final_flat.reshape(b, c, self.period_len, self.seg_num_y)
        
        # Segment分支处理
        # Step 2a: Segment分支独立归一化
        x_segment_flat = x_segment_branch.reshape(-1, self.period_len, self.seg_num_x)
        x_segment_norm, segment_norm_stats = self._normalize_for_segment_branch(x_segment_flat, b, c)
        x_segment_branch = x_segment_norm.reshape(b, c, self.period_len, self.seg_num_x)
        
        # Step 2b: Segment降维
        x_segment, segment_basis_combined = self._apply_segment_reduction(x_segment_branch, b, c)
        
        # Step 2c: Segment分支逆归一化
        x_segment_final_flat = x_segment.reshape(-1, self.period_len, self.seg_num_y)
        x_segment_final_flat = self._denormalize_for_segment_branch(x_segment_final_flat, segment_norm_stats, b, c)
        x_segment_final = x_segment_final_flat.reshape(b, c, self.period_len, self.seg_num_y)
        
        # Step 3: 融合两个分支的结果
        x_fused = x_segment_final + x_period_final
        
        return x_fused, segment_basis_combined, period_basis_combined

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        前向传播：并行降维策略（Period和Segment并行，分支独立归一化）
        """
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

        # 并行降维策略 - 内存布局优化版本:（内部处理归一化和逆归一化）
        x_4d = x.reshape(b, c, self.period_len, self.seg_num_x)
        x_parallel_reduced, segment_basis, period_basis = self._apply_parallel_reduction(x_4d, b, c)

        # Reshape back - 使用优化后的结果
        x = x_parallel_reduced.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y)
        x = x.permute(0, 1, 3, 2)  # (batch_size, enc_in, seg_num_y, period_len)
        x = x.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)  # b t c

        # 输出处理
        output = x[:, : self.pred_len, :]

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
            
            # Segment basis的正交损失
            if segment_basis is not None:
                segment_orthogonal_loss = cal_orthogonal_loss(segment_basis)
                weighted_segment_loss = self.segment_ortho_weight * segment_orthogonal_loss
                total_orthogonal_loss += weighted_segment_loss
                self.log("train_loss_segment_orthogonal", segment_orthogonal_loss, on_epoch=True)
                self.log("train_loss_segment_weighted", weighted_segment_loss, on_epoch=True)
            
            # Period basis的正交损失
            if period_basis is not None:
                period_orthogonal_loss = cal_orthogonal_loss(period_basis)
                weighted_period_loss = self.period_ortho_weight * period_orthogonal_loss
                total_orthogonal_loss += weighted_period_loss
                self.log("train_loss_period_orthogonal", period_orthogonal_loss, on_epoch=True)
                self.log("train_loss_period_weighted", weighted_period_loss, on_epoch=True)
            
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
