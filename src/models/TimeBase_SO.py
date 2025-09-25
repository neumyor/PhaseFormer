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


class TimeBase_SO(DefaultPLModule):
    """
    TimeBase Segment Only (SO) 模型：仅使用Segment降维的时间序列预测模型
    
    策略特点：
    - 仅使用Segment降维，发现相似segment间的冗余性
    - 学习跨周期的时序模式
    - 计算复杂度最低，内存使用最少
    - 兼容period_norm和全局归一化
    - 不支持segment_norm（segment_norm设计用于period_reduction策略）
    
    适用场景：
    - segment间相似度高的时间序列
    - 计算资源有限的环境
    - 需要快速训练和推理的场景
    """
    def __init__(self, configs):
        super(TimeBase_SO, self).__init__(configs)
        
        # 基础参数
        self.use_period_norm = configs.use_period_norm
        self.use_segment_norm = getattr(configs, "use_segment_norm", False)
        self.use_orthogonal = configs.use_orthogonal
        
        # 正交权重配置
        self.orthogonal_weight = getattr(configs, "orthogonal_weight", 0.04)
        self.segment_ortho_weight = getattr(configs, "segment_ortho_weight", self.orthogonal_weight)
        
        # 模型参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.pad_seq_len = 0
        self.basis_num = configs.basis_num
        self.individual = configs.individual
        
        # Segment Only策略的参数验证
        if self.use_period_norm:
            raise ValueError(
                "TimeBase_SO (segment_only) is not compatible with period_norm. "
                "Period_norm is designed for strategies that use period_reduction. "
                "Please set use_period_norm=False for TimeBase_SO."
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
        """设置Segment降维的线性层"""
        if self.individual:
            # Individual模式：为每个变量创建独立的线性层
            self.segment_ts2basis = nn.ModuleList()
            self.segment_basis2ts = nn.ModuleList()
            for i in range(self.enc_in):
                self.segment_ts2basis.append(
                    nn.Linear(self.seg_num_x, self.basis_num)
                )
                self.segment_basis2ts.append(
                    nn.Linear(self.basis_num, self.seg_num_y)
                )
        else:
            # Shared模式：所有变量共享线性层
            self.segment_ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
            self.segment_basis2ts = nn.Linear(self.basis_num, self.seg_num_y)

    def _apply_segment_reduction(self, x, b, c):
        """
        应用Segment维度降维 (enc_in支持tensor并行)
        
        输入: x (b, c, period_len, seg_num_x)
        输出: x_transformed (b, c, period_len, seg_num_y), segment_basis_combined
        """
        if self.individual:
            # Individual模式：为每个变量创建独立的线性层
            # x: (b, c, period_len, seg_num_x)
            # 先reshape合并batch和period_len，方便并行
            x_reshape = x.permute(1, 0, 2, 3)  # (c, b, period_len, seg_num_x)
            x_reshape = x_reshape.reshape(self.enc_in, -1, self.seg_num_x)  # (c, b*period_len, seg_num_x)
            # 并行处理每个变量
            segment_basis_list = []
            segment_pred_list = []
            for i in range(self.enc_in):
                seg_basis = self.segment_ts2basis[i](x_reshape[i])  # (b*period_len, basis_num)
                seg_pred = self.segment_basis2ts[i](seg_basis)      # (b*period_len, seg_num_y)
                segment_basis_list.append(seg_basis)
                segment_pred_list.append(seg_pred)
            # 堆叠回去
            segment_basis = torch.stack(segment_basis_list, dim=0)  # (c, b*period_len, basis_num)
            segment_pred = torch.stack(segment_pred_list, dim=0)    # (c, b*period_len, seg_num_y)
            # reshape回原始形状
            segment_basis = segment_basis.reshape(self.enc_in, b, self.period_len, self.basis_num).permute(1,0,2,3)  # (b, c, period_len, basis_num)
            segment_pred = segment_pred.reshape(self.enc_in, b, self.period_len, self.seg_num_y).permute(1,0,2,3)    # (b, c, period_len, seg_num_y)
        else:
            # Shared模式：所有变量共享线性层
            # x: (b, c, period_len, seg_num_x)
            x_reshape = x.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
            segment_basis = self.segment_ts2basis(x_reshape)  # (b*c*period_len, basis_num)
            segment_pred = self.segment_basis2ts(segment_basis)  # (b*c*period_len, seg_num_y)
            # reshape回原始形状
            segment_basis = segment_basis.reshape(b, c, self.period_len, self.basis_num)
            segment_pred = segment_pred.reshape(b, c, self.period_len, self.seg_num_y)

        # segment_basis_combined: (b, c, period_len, basis_num) -> (-1, period_len, basis_num)
        segment_basis_combined = segment_basis.reshape(-1, self.period_len, self.basis_num)
        return segment_pred, segment_basis_combined

    def _normalize_input(self, x, b, c):
        """
        输入归一化：支持segment_norm和全局归一化
        """
        if self.use_segment_norm:
            # Segment归一化：对每个segment内的所有period求均值
            # x: (bc, period_len, seg_num_x)，在seg_num维度求均值
            segment_mean = torch.mean(x, dim=-1, keepdim=True)  # (bc, period_len, 1)
            x = x - segment_mean
            return x, {"segment_mean": segment_mean}
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
        if self.use_segment_norm:
            # Segment归一化：直接加回segment_mean
            segment_mean = norm_stats["segment_mean"]
            x = x + segment_mean
        else:
            # 全局归一化：reshape，加mean，再reshape回来
            x = x.reshape(b, c, -1)
            x = x + norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        前向传播：Segment Only策略
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

        # Normalize
        x, norm_stats = self._normalize_input(x, b, c)

        # Segment降维
        x = x.reshape(b, c, self.period_len, self.seg_num_x)
        x, segment_basis = self._apply_segment_reduction(x, b, c)

        # Denormalize
        x = x.reshape(-1, self.period_len, self.seg_num_y)
        x = self._denormalize_output(x, norm_stats, b, c)

        # Reshape back
        x = x.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y)
        x = x.permute(0, 1, 3, 2)  # (batch_size, enc_in, seg_num_y, period_len)
        x = x.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)  # b t c

        # 输出处理
        output = x[:, : self.pred_len, :]

        # 如果需要正交损失，返回segment_basis
        if self.use_orthogonal:
            return output, segment_basis
        else:
            return output

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = self._build_decoder_input(batch_y)

        # 调用forward方法
        forward_output = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        # 处理forward方法的返回值
        if self.use_orthogonal:
            outputs, segment_basis = forward_output
        else:
            outputs = forward_output

        outputs = outputs[:, -self.args.dataset_args.pred_len :, :]
        batch_y = batch_y[:, -self.args.dataset_args.pred_len :, :]

        if self.target_var_index != -1:
            batch_y = batch_y[:, :, self.target_var_index].unsqueeze(-1)

        criterion = self._get_criterion(self.args.training_args.loss_func)
        loss = criterion(outputs, batch_y)

        # 正交损失计算
        if self.use_orthogonal and segment_basis is not None:
            segment_orthogonal_loss = cal_orthogonal_loss(segment_basis)
            weighted_segment_loss = self.segment_ortho_weight * segment_orthogonal_loss
            total_loss = loss + weighted_segment_loss
            
            # 记录损失
            self.log("train_loss_main", loss, on_epoch=True)
            self.log("train_loss_segment_orthogonal", segment_orthogonal_loss, on_epoch=True)
            self.log("train_loss_segment_weighted", weighted_segment_loss, on_epoch=True)
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
            outputs, _ = forward_output  # 忽略basis
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
            outputs, _ = forward_output  # 忽略basis
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
