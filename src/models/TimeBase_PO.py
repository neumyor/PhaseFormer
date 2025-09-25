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


class TimeBase_PO(DefaultPLModule):
    """
    TimeBase Period Only (PO) 模型：仅使用Period降维的时间序列预测模型
    
    策略特点：
    - 仅使用Period降维，基于延迟嵌入冗余性
    - 学习不同period位置的延迟嵌入间的相关性
    - 计算复杂度中等，内存使用中等
    - 必须使用全局归一化（不兼容period_norm）
    - 兼容segment_norm
    
    适用场景：
    - 周期性很强但segment间差异大的时间序列
    - period内部结构有冗余的数据
    - 需要捕获period内模式的场景
    """
    def __init__(self, configs):
        super(TimeBase_PO, self).__init__(configs)
        
        # 基础参数
        self.use_period_norm = configs.use_period_norm
        self.use_segment_norm = getattr(configs, "use_segment_norm", False)
        self.use_orthogonal = configs.use_orthogonal
        
        # 正交权重配置
        self.orthogonal_weight = getattr(configs, "orthogonal_weight", 0.04)
        self.period_ortho_weight = getattr(configs, "period_ortho_weight", self.orthogonal_weight)
        
        # 模型参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.pad_seq_len = 0
        self.basis_num = configs.basis_num
        self.period_basis_num = getattr(configs, "period_basis_num", self.period_len)
        self.individual = configs.individual
        
        # Period Only策略的参数验证
        if self.use_segment_norm:
            raise ValueError(
                "TimeBase_PO (period_only) is not compatible with segment_norm. "
                "Segment_norm is designed for strategies that use segment_reduction. "
                "Please set use_segment_norm=False for TimeBase_PO."
            )
        
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
        """设置Period降维的线性层"""
        if self.individual:
            # Individual模式：为每个变量创建独立的线性层
            self.period_ts2basis = nn.ModuleList()
            self.period_basis2ts = nn.ModuleList()
            self.period_output_mapping = nn.ModuleList()
            
            for i in range(self.enc_in):
                # Period降维层
                self.period_ts2basis.append(
                    nn.Linear(self.period_len, self.period_basis_num)
                )
                self.period_basis2ts.append(
                    nn.Linear(self.period_basis_num, self.period_len)
                )
                # 输出映射层：seg_num_x -> seg_num_y
                self.period_output_mapping.append(
                    nn.Linear(self.seg_num_x, self.seg_num_y)
                )
        else:
            # Shared模式：所有变量共享线性层
            self.period_ts2basis = nn.Linear(self.period_len, self.period_basis_num)
            self.period_basis2ts = nn.Linear(self.period_basis_num, self.period_len)
            self.period_output_mapping = nn.Linear(self.seg_num_x, self.seg_num_y)

    def _apply_period_reduction(self, x, b, c):
        """
        应用Period维度降维 - 优化版本：批量处理减少循环
        
        输入: x (b, c, period_len, seg_num_x)
        输出: x_transformed (b, c, period_len, seg_num_x), period_basis_combined
        """
        if self.individual:
            # Individual模式：批量处理所有变量
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
            
            # 批量处理所有数据
            period_coeffs = self.period_ts2basis(x_flat)  # (b*c*seg_num_x, period_basis_num)
            period_basis = period_coeffs.reshape(b, c, self.seg_num_x, self.period_basis_num)
            
            period_reconstructed = self.period_basis2ts(period_coeffs)  # (b*c*seg_num_x, period_len)
            x_reconstructed = period_reconstructed.reshape(b, c, self.seg_num_x, self.period_len)
            x_output = x_reconstructed.permute(0, 1, 3, 2)  # (b, c, period_len, seg_num_x)
            
            period_basis_combined = period_basis.reshape(-1, self.seg_num_x, self.period_basis_num)
        
        return x_output, period_basis_combined

    def _apply_period_only_reduction(self, x, b, c):
        """
        应用仅Period降维策略 - 优化版本：向量化映射
        
        输入: x (b, c, period_len, seg_num_x)
        输出: x_transformed (b, c, period_len, seg_num_y), period_basis_combined
        """
        # Step 1: Period降维
        x_period, period_basis_combined = self._apply_period_reduction(x, b, c)
        
        # Step 2: 向量化映射到目标segment数量 (seg_num_x -> seg_num_y)
        if self.individual:
            # Individual模式：批量处理所有变量
            x_reshaped = x_period.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_x)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_x)  # (c, b*period_len, seg_num_x)
            
            x_output_list = []
            for i in range(self.enc_in):
                period_mapped = self.period_output_mapping[i](x_flat[i])  # (b*period_len, seg_num_y)
                period_reshaped = period_mapped.reshape(b, self.period_len, self.seg_num_y)
                x_output_list.append(period_reshaped)
            
            x_output = torch.stack(x_output_list, dim=1)  # (b, c, period_len, seg_num_y)
        else:
            # Shared模式：完全向量化处理
            x_flat = x_period.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
            period_mapped = self.period_output_mapping(x_flat)  # (b*c*period_len, seg_num_y)
            x_output = period_mapped.reshape(b, c, self.period_len, self.seg_num_y)
        
        return x_output, period_basis_combined

    def _normalize_input(self, x, b, c):
        """
        输入归一化：支持period_norm和全局归一化
        """
        if self.use_period_norm:
            # Period归一化：对每个period位置的所有segment求均值
            # x: (bc, period_len, seg_num_x)，在period_len维度求均值
            period_mean = torch.mean(x, dim=1, keepdim=True)  # (bc, 1, seg_num_x)
            x = x - period_mean
            return x, {"period_mean": period_mean}
        else:
            # 全局归一化：对每个变量的所有数据求均值
            x = x.reshape(b, c, -1)
            mean = torch.mean(x, dim=-1, keepdim=True)  # (b, c, 1)
            x = x - mean
            x = x.reshape(-1, self.period_len, self.seg_num_x)
            return x, {"mean": mean}

    def _denormalize_output(self, x, norm_stats, b, c):
        """
        输出逆归一化
        """
        if self.use_period_norm:
            # Period归一化的逆操作
            period_mean = norm_stats["period_mean"]  # (bc, 1, seg_num_x)
            
            if self.seg_num_x == self.seg_num_y:
                # segment数量相同，直接加回
                x = x + period_mean
            else:
                # segment数量不同，需要插值
                # 将period_mean从(bc, 1, seg_num_x)插值到(bc, 1, seg_num_y)
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
            # 全局归一化：reshape，加mean，再reshape回来
            x = x.reshape(b, c, -1)
            x = x + norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        前向传播：Period Only策略
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

        # Period Only降维策略 - 内存布局优化版本:
        # 归一化
        x, norm_stats = self._normalize_input(x, b, c)

        # Period Only降维 - 保持高效的张量布局
        x_4d = x.reshape(b, c, self.period_len, self.seg_num_x)
        x_period_reduced, period_basis = self._apply_period_only_reduction(x_4d, b, c)

        # 逆归一化 - 直接在4D张量上操作
        x_final_flat = x_period_reduced.reshape(-1, self.period_len, self.seg_num_y)
        x_denormed = self._denormalize_output(x_final_flat, norm_stats, b, c)

        # Reshape back - 使用最终处理的结果
        x = x_denormed.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y)
        x = x.permute(0, 1, 3, 2)  # (batch_size, enc_in, seg_num_y, period_len)
        x = x.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)  # b t c

        # 输出处理
        output = x[:, : self.pred_len, :]

        # 如果需要正交损失，返回period_basis
        if self.use_orthogonal:
            return output, period_basis
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
            outputs, period_basis = forward_output
        else:
            outputs = forward_output

        outputs = outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        loss = criterion(outputs, batch_y)

        # 正交损失计算
        if self.use_orthogonal and period_basis is not None:
            period_orthogonal_loss = cal_orthogonal_loss(period_basis)
            weighted_period_loss = self.period_ortho_weight * period_orthogonal_loss
            total_loss = loss + weighted_period_loss
            
            # 记录损失
            self.log("train_loss_main", loss, on_epoch=True)
            self.log("train_loss_period_orthogonal", period_orthogonal_loss, on_epoch=True)
            self.log("train_loss_period_weighted", weighted_period_loss, on_epoch=True)
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
            outputs, _ = forward_output
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
            outputs, _ = forward_output
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
