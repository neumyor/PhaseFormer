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


class TimeBase_SP(DefaultPLModule):
    """
    TimeBase Segment-Period Sequential (SP) 模型：串行双维度降维的时间序列预测模型（PS的逆转版本）
    
    策略特点：
    - 使用Segment和Period双维度降维，串行执行（与PS相反的顺序）
    - 先Segment降维，再Period降维
    - 支持精细的分阶段归一化策略
    - 计算复杂度高，内存使用高
    - 支持segment_norm和period_norm的组合使用
    
    归一化流程（与PS相反）：
    1. Segment reduction前：segment_norm（如果启用）
    2. Segment reduction后：segment_denorm + period_norm（如果启用）
    3. Period reduction后：period_denorm（如果启用）
    
    这种设计允许两种归一化都发挥作用：
    - segment_norm帮助segment_reduction更好地学习segment间模式
    - period_norm帮助period_reduction更好地学习period内模式
    
    适用场景：
    - Segment和Period都有冗余，但更适合先处理segment模式
    - 数据中segment间的相似性比period内的模式更重要
    - 需要先捕获跨周期模式，再细化period内结构
    - 追求与PS不同的特征提取顺序
    """
    def __init__(self, configs):
        super(TimeBase_SP, self).__init__(configs)
        
        # 基础参数
        self.use_period_norm = configs.use_period_norm
        self.use_segment_norm = getattr(configs, "use_segment_norm", False)
        self.use_orthogonal = configs.use_orthogonal
        
        # 正交权重配置（与PS相反，segment权重更高）
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
        
        # 设置线性层
        self._setup_linear_layers()

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
                # 这是关键的形状处理层
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

    def _apply_segment_reduction(self, x, b, c):
        """
        应用Segment维度降维 - 优化版本：批量处理减少循环
        
        输入: x (b, c, period_len, seg_num_x)
        输出: x_output (b, c, period_len, seg_num_y), segment_basis_combined
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

    def _apply_period_reduction_with_mapping(self, x, b, c):
        """
        应用Period维度降维 - 带有中间映射层处理维度变化
        
        输入: x (b, c, period_len, seg_num_y)  # 注意：这里是seg_num_y
        输出: x_output (b, c, period_len, seg_num_y), period_basis_combined
        """
        # Step 1: 先通过映射层将seg_num_y映射回seg_num_x进行period降维
        if self.individual:
            # Individual模式：批量处理所有变量的映射
            x_reshaped = x.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_y)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_y)  # (c, b*period_len, seg_num_y)
            
            x_mapped_list = []
            for i in range(self.enc_in):
                # 映射到seg_num_x以便进行period降维
                x_mapped = self.segment_to_period_mapping[i](x_flat[i])  # (b*period_len, seg_num_x)
                x_mapped_reshaped = x_mapped.reshape(b, self.period_len, self.seg_num_x)
                x_mapped_list.append(x_mapped_reshaped)
            
            x_mapped = torch.stack(x_mapped_list, dim=1)  # (b, c, period_len, seg_num_x)
        else:
            # Shared模式：完全向量化映射
            x_flat = x.reshape(-1, self.seg_num_y)  # (b*c*period_len, seg_num_y)
            x_mapped_flat = self.segment_to_period_mapping(x_flat)  # (b*c*period_len, seg_num_x)
            x_mapped = x_mapped_flat.reshape(b, c, self.period_len, self.seg_num_x)
        
        # Step 2: Period降维处理
        if self.individual:
            # Individual模式：批量处理所有变量的Period降维
            x_transposed = x_mapped.permute(0, 1, 3, 2)  # (b, c, seg_num_x, period_len)
            x_reshaped = x_transposed.permute(1, 0, 2, 3)  # (c, b, seg_num_x, period_len)
            x_flat = x_reshaped.reshape(c, -1, self.period_len)  # (c, b*seg_num_x, period_len)
            
            # 批量处理所有变量
            period_basis_list = []
            x_period_output_list = []
            
            for i in range(self.enc_in):
                # 获取系数并重建
                period_coeffs = self.period_ts2basis[i](x_flat[i])  # (b*seg_num_x, period_basis_num)
                period_basis = period_coeffs.reshape(b, self.seg_num_x, self.period_basis_num)
                
                period_reconstructed = self.period_basis2ts[i](period_coeffs)  # (b*seg_num_x, period_len)
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
            # Individual模式：批量处理最终映射
            x_reshaped = x_period_output.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_x)
            x_flat = x_reshaped.reshape(c, -1, self.seg_num_x)  # (c, b*period_len, seg_num_x)
            
            x_final_list = []
            for i in range(self.enc_in):
                # 这里我们重用segment映射层的逆操作（通过linear近似）
                # 实际上应该是seg_num_x -> seg_num_y的映射
                # 由于线性层是可逆的，我们可以学习这个映射
                x_final = self.segment_basis2ts[i](
                    self.segment_ts2basis[i](x_flat[i])
                )  # (b*period_len, seg_num_y)
                x_final_reshaped = x_final.reshape(b, self.period_len, self.seg_num_y)
                x_final_list.append(x_final_reshaped)
            
            x_output = torch.stack(x_final_list, dim=1)  # (b, c, period_len, seg_num_y)
        else:
            # Shared模式：完全向量化最终映射
            x_flat = x_period_output.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
            x_final_flat = self.segment_basis2ts(
                self.segment_ts2basis(x_flat)
            )  # (b*c*period_len, seg_num_y)
            x_output = x_final_flat.reshape(b, c, self.period_len, self.seg_num_y)
        
        return x_output, period_basis_combined

    def _apply_segment_norm(self, x, b, c):
        """
        阶段1：Segment reduction前的segment归一化
        """
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

    def _apply_segment_denorm_and_period_norm(self, x, segment_norm_stats, b, c):
        """
        阶段2：Segment reduction后的segment逆归一化 + period归一化
        注意：此时x是(bc, period_len, seg_num_y)维度
        """
        # Step 1: Segment逆归一化
        if self.use_segment_norm and "segment_mean" in segment_norm_stats:
            segment_mean = segment_norm_stats["segment_mean"]  # (bc, period_len, 1)
            # segment_mean可以直接广播到任何segment维度上
            x = x + segment_mean
        elif "mean" in segment_norm_stats:
            # 全局归一化的逆操作
            x = x.reshape(b, c, -1)
            x = x + segment_norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)

        # Step 2: Period归一化
        if self.use_period_norm:
            # Period归一化：在period维度求均值
            # x: (bc, period_len, seg_num_y)，在period维度求均值
            period_mean = torch.mean(x, dim=1, keepdim=True)  # (bc, 1, seg_num_y)
            x = x - period_mean
            return x, {"period_mean": period_mean}
        else:
            # 不做period归一化，直接返回
            return x, {}

    def _apply_period_denorm(self, x, period_norm_stats, b, c):
        """
        阶段3：Period reduction后的period逆归一化
        """
        if self.use_period_norm and "period_mean" in period_norm_stats:
            # Period归一化的逆操作
            # x的形状：(bc, period_len, seg_num_y)
            period_mean = period_norm_stats["period_mean"]  # (bc, 1, seg_num_y)
            x = x + period_mean
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        前向传播：串行降维策略（先Segment，再Period）- PS的逆转版本
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

        # 分阶段归一化串行降维策略 - SP版本（与PS相反的顺序）:
        # 阶段1: Segment reduction前的segment归一化
        x, segment_norm_stats = self._apply_segment_norm(x, b, c)
        
        # 阶段1: Segment降维 - 保持高效的张量布局
        x_4d = x.reshape(b, c, self.period_len, self.seg_num_x)
        x_segment_reduced, segment_basis = self._apply_segment_reduction(x_4d, b, c)
        
        # 阶段2: Segment reduction后的segment逆归一化 + period归一化
        # 直接在4D张量上操作，避免flatten和reshape循环
        x_flat = x_segment_reduced.reshape(-1, self.period_len, self.seg_num_y)
        x_period_normed, period_norm_stats = self._apply_segment_denorm_and_period_norm(x_flat, segment_norm_stats, b, c)
        
        # 阶段2: Period降维 - 重用4D布局，优化的批量处理
        x_4d_normed = x_period_normed.reshape(b, c, self.period_len, self.seg_num_y)
        x_period_reduced, period_basis = self._apply_period_reduction_with_mapping(x_4d_normed, b, c)

        # 阶段3: Period reduction后的period逆归一化
        x_final_flat = x_period_reduced.reshape(-1, self.period_len, self.seg_num_y)
        x_denormed = self._apply_period_denorm(x_final_flat, period_norm_stats, b, c)

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

        # 正交损失计算（分层权重，与PS不同，segment权重更高）
        if self.use_orthogonal:
            total_orthogonal_loss = 0
            
            # Segment basis的正交损失（SP中segment是主要的）
            if segment_basis is not None:
                segment_orthogonal_loss = cal_orthogonal_loss(segment_basis)
                weighted_segment_loss = self.segment_ortho_weight * segment_orthogonal_loss
                total_orthogonal_loss += weighted_segment_loss
                self.log("train_loss_segment_orthogonal", segment_orthogonal_loss, on_epoch=True)
                self.log("train_loss_segment_weighted", weighted_segment_loss, on_epoch=True)
            
            # Period basis的正交损失（SP中period是辅助的）
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
