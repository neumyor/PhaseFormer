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


class TimeBase_PS(DefaultPLModule):
    """
    TimeBase Period-Segment Sequential (PS) 模型：串行双维度降维的时间序列预测模型
    
    策略特点：
    - 使用Period和Segment双维度降维，串行执行
    - 支持精细的分阶段归一化策略
    - 计算复杂度高，内存使用高
    - 支持period_norm和segment_norm的组合使用
    
    归一化流程：
    1. Period reduction前：period_norm（如果启用）
    2. Period reduction后：period_denorm + segment_norm（如果启用）
    3. Segment reduction后：segment_denorm（如果启用）
    
    这种设计允许两种归一化都发挥作用：
    - period_norm帮助period_reduction更好地学习period内模式
    - segment_norm帮助segment_reduction更好地学习segment间模式
    
    适用场景：
    - Period和Segment都有冗余，需要逐步压缩
    - 数据具有明显的层次结构
    - 需要对不同降维阶段使用不同归一化策略
    - 追求较好性能且能接受高计算开销
    """
    def __init__(self, configs):
        super(TimeBase_PS, self).__init__(configs)
        
        # 基础参数
        self.use_period_norm = configs.use_period_norm
        self.use_segment_norm = getattr(configs, "use_segment_norm", False)
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
        
        # Sequential策略的参数验证
        # PS策略现在支持精细的双阶段归一化：
        # 1. Period reduction前：period norm
        # 2. Period reduction后：period denorm + segment norm  
        # 3. Segment reduction后：segment denorm
        # 这样可以让两种归一化都发挥作用
        
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
        """设置Period和Segment降维的线性层"""
        if self.individual:
            # Individual模式：为每个变量创建独立的线性层
            # Period降维层
            self.period_ts2basis = nn.ModuleList()
            self.period_basis2ts = nn.ModuleList()
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
            # Segment降维层
            self.segment_ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
            self.segment_basis2ts = nn.Linear(self.basis_num, self.seg_num_y)

    def _apply_period_reduction(self, x, b, c):
        """
        应用Period维度降维 - 基于延迟嵌入冗余性
        注意：Period reduction后需要保持seg_num_x，segment reduction时才映射到seg_num_y
        """
        if self.individual:
            x_period_basis_list = []
            # 直接在原张量上进行操作，避免创建zeros_like
            x_output = x.clone()
            
            for i in range(self.enc_in):
                period_data = x[:, i, :, :]  # (b, period_len, seg_num_x)
                
                # 获取period basis系数
                period_data_transposed = period_data.permute(0, 2, 1)  # (b, seg_num_x, period_len)
                period_data_flat = period_data_transposed.reshape(-1, self.period_len)
                
                period_coeffs = self.period_ts2basis[i](period_data_flat)
                period_basis = period_coeffs.reshape(b, self.seg_num_x, self.period_basis_num)
                
                # 重建
                period_reconstructed = self.period_basis2ts[i](period_coeffs)
                period_pred_transposed = period_reconstructed.reshape(b, self.seg_num_x, self.period_len)
                period_pred = period_pred_transposed.permute(0, 2, 1)
                
                x_output[:, i, :, :] = period_pred
                x_period_basis_list.append(period_basis)
            
            period_basis_combined = torch.stack(x_period_basis_list, dim=1)
            period_basis_combined = period_basis_combined.reshape(-1, self.seg_num_x, self.period_basis_num)
            x = x_output
        else:
            x_transposed = x.permute(0, 1, 3, 2)  # (b, c, seg_num_x, period_len)
            x_flat = x_transposed.reshape(-1, self.period_len)
            
            period_coeffs = self.period_ts2basis(x_flat)
            period_basis = period_coeffs.reshape(b, c, self.seg_num_x, self.period_basis_num)
            
            period_reconstructed = self.period_basis2ts(period_coeffs)
            x_reconstructed = period_reconstructed.reshape(b, c, self.seg_num_x, self.period_len)
            period_pred = x_reconstructed.permute(0, 1, 3, 2)
            
            x = period_pred
            period_basis_combined = period_basis.reshape(-1, self.seg_num_x, self.period_basis_num)
        
        return x, period_basis_combined

    def _apply_segment_reduction(self, x, b, c):
        """
        应用Segment维度降维 - 优化版本：批量处理减少循环
        """
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

    def _apply_period_norm(self, x, b, c):
        """
        阶段1：Period reduction前的period归一化
        """
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

    def _apply_period_denorm_and_segment_norm(self, x, period_norm_stats, b, c):
        """
        阶段2：Period reduction后的period逆归一化 + segment归一化
        注意：此时x仍然是seg_num_x维度，period_mean也是seg_num_x维度，可以直接操作
        """
        # Step 1: Period逆归一化
        if self.use_period_norm and "period_mean" in period_norm_stats:
            period_mean = period_norm_stats["period_mean"]  # (bc, 1, seg_num_x)
            # 此时x的形状是(bc, period_len, seg_num_x)，period_mean是(bc, 1, seg_num_x)
            # 可以直接广播相加
            x = x + period_mean
        elif "mean" in period_norm_stats:
            # 全局归一化的逆操作
            x = x.reshape(b, c, -1)
            x = x + period_norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_x)

        # Step 2: Segment归一化
        if self.use_segment_norm:
            # Segment归一化：在seg_num维度求均值
            # x: (bc, period_len, seg_num_x)，在seg_num维度求均值
            segment_mean = torch.mean(x, dim=-1, keepdim=True)  # (bc, period_len, 1)
            x = x - segment_mean
            return x, {"segment_mean": segment_mean}
        else:
            # 不做segment归一化，直接返回
            return x, {}

    def _apply_segment_denorm(self, x, segment_norm_stats, b, c):
        """
        阶段3：Segment reduction后的segment逆归一化
        注意：segment_mean是在seg_num_x维度计算的，但现在x是seg_num_y维度
        由于segment_mean是对所有segment求均值得到的(bc, period_len, 1)，
        它可以直接广播到任何segment维度上
        """
        if self.use_segment_norm and "segment_mean" in segment_norm_stats:
            # Segment归一化的逆操作
            # x的形状：(bc, period_len, seg_num_y)
            segment_mean = segment_norm_stats["segment_mean"]  # (bc, period_len, 1)
            x = x + segment_mean
        return x



    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        前向传播：串行降维策略（先Period，再Segment）
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

        # 分阶段归一化串行降维策略 - 内存布局优化版本:
        # 阶段1: Period reduction前的period归一化
        x, period_norm_stats = self._apply_period_norm(x, b, c)
        
        # 阶段1: Period降维 - 保持高效的张量布局
        x_4d = x.reshape(b, c, self.period_len, self.seg_num_x)
        x_period_reduced, period_basis = self._apply_period_reduction(x_4d, b, c)
        
        # 阶段2: Period reduction后的period逆归一化 + segment归一化
        # 直接在4D张量上操作，避免flatten和reshape循环
        x_flat = x_period_reduced.reshape(-1, self.period_len, self.seg_num_x)
        x_segment_normed, segment_norm_stats = self._apply_period_denorm_and_segment_norm(x_flat, period_norm_stats, b, c)
        
        # 阶段2: Segment降维 - 重用4D布局，优化的批量处理
        x_4d_normed = x_segment_normed.reshape(b, c, self.period_len, self.seg_num_x)
        x_segment_reduced, segment_basis = self._apply_segment_reduction(x_4d_normed, b, c)

        # 阶段3: Segment reduction后的segment逆归一化
        x_final_flat = x_segment_reduced.reshape(-1, self.period_len, self.seg_num_y)
        x_denormed = self._apply_segment_denorm(x_final_flat, segment_norm_stats, b, c)

        # Reshape back - 使用最终处理的结果
        x = x_denormed.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y)
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
