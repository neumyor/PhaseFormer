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


class TimeBase(DefaultPLModule):
    """
    TimeBase模型：支持多种降维策略的时间序列预测模型
    
    支持的降维策略（通过reduction_strategy参数控制）：
    1. 'segment_only': 仅Segment降维
       - use_segment_reduction=True, use_period_reduction=False
       - 兼容period_norm=True
    2. 'period_only': 仅Period降维
       - use_segment_reduction=False, use_period_reduction=True
       - 必须使用全局归一化（period_norm=False）
    3. 'sequential': 串行双维度降维
       - use_segment_reduction=True, use_period_reduction=True
       - 先Period降维，再Segment降维，必须period_norm=False
    4. 'parallel': 并行双维度降维
       - use_segment_reduction=True, use_period_reduction=True  
       - Period和Segment降维并行执行，结果融合，必须period_norm=False
    
    降维设计理念：
    - Segment降维：发现相似segment间的冗余性，学习跨周期的时序模式
    - Period降维：基于延迟嵌入冗余性，学习不同period位置的延迟嵌入间的相关性
      * 每个period位置t对应一个延迟嵌入 [seg1[t], seg2[t], ..., segN[t]]
      * 用period_basis_num个基表示period_len个延迟嵌入
    
    策略特点：
    - segment_only: 最简单，仅利用segment间的时序模式相似性，支持period归一化
    - period_only: 仅利用period内的延迟嵌入冗余性，适合周期性强但segment差异大的数据
    - sequential: 串行处理，先压缩period维度，再学习segment模式
    - parallel: 并行处理，两个维度独立建模后融合，可能学到更丰富的特征
    
    归一化兼容性：
    - segment_only: 兼容period_norm=True/False，不兼容segment_norm
    - period_only: 必须period_norm=False，兼容segment_norm=True/False
    - sequential: 必须period_norm=False，兼容segment_norm=True/False
    - parallel: 必须period_norm=False，兼容segment_norm=True/False，支持分支独立归一化
    
    新增归一化策略：
    - segment_norm: 在segment维度上归一化，适用于period_reduction策略
    
    正交损失：分别应用于Segment basis和Period basis（如果存在）
    """
    def __init__(self, configs):
        super(TimeBase, self).__init__(configs)
        self.use_period_norm = configs.use_period_norm
        self.use_segment_norm = getattr(configs, "use_segment_norm", False)  # 新增segment归一化
        self.use_orthogonal = configs.use_orthogonal
        
        # 分层权重配置
        self.orthogonal_weight = getattr(configs, "orthogonal_weight", 0.04)
        self.segment_ortho_weight = getattr(configs, "segment_ortho_weight", self.orthogonal_weight)
        self.period_ortho_weight = getattr(configs, "period_ortho_weight", self.orthogonal_weight * 0.1)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.pad_seq_len = 0
        self.basis_num = configs.basis_num

        # 新增：period维度降维参数
        self.period_basis_num = getattr(
            configs, "period_basis_num", self.period_len
        )  # 默认不降维

        # 新增：降维映射开关
        self.use_segment_reduction = getattr(
            configs, "use_segment_reduction", True
        )  # segment维度降维
        self.use_period_reduction = getattr(
            configs, "use_period_reduction", False
        )  # period维度降维
        
        # 新增：降维策略控制
        self.reduction_strategy = getattr(
            configs, "reduction_strategy", "sequential"
        )  # 'sequential', 'parallel', 'segment_only'

        # 参数验证：支持四种策略
        if self.reduction_strategy == "segment_only":
            if self.use_period_reduction:
                raise ValueError(
                    "reduction_strategy='segment_only' conflicts with use_period_reduction=True. "
                    "Please set use_period_reduction=False or use other strategies."
                )
            self.use_segment_reduction = True
            self.use_period_reduction = False
        elif self.reduction_strategy == "period_only":
            if self.use_segment_reduction:
                raise ValueError(
                    "reduction_strategy='period_only' conflicts with use_segment_reduction=True. "
                    "Please set use_segment_reduction=False or use other strategies."
                )
            self.use_segment_reduction = False
            self.use_period_reduction = True
        elif self.reduction_strategy == "sequential":
            if not (self.use_segment_reduction and self.use_period_reduction):
                raise ValueError(
                    "reduction_strategy='sequential' requires both use_segment_reduction=True "
                    "and use_period_reduction=True."
                )
        elif self.reduction_strategy == "parallel":
            if not (self.use_segment_reduction and self.use_period_reduction):
                raise ValueError(
                    "reduction_strategy='parallel' requires both use_segment_reduction=True "
                    "and use_period_reduction=True."
                )
        else:
            raise ValueError(
                f"Invalid reduction_strategy: {self.reduction_strategy}. "
                "Supported strategies: 'segment_only', 'period_only', 'sequential', 'parallel'."
            )

        # 参数兼容性检查 - 根据不同策略调整normalization规则
        if self.reduction_strategy == "segment_only":
            # segment_only策略不能使用segment_norm
            if self.use_segment_norm:
                raise ValueError(
                    "segment_only strategy is not compatible with segment_norm. "
                    "Segment_norm is designed for strategies that use period_reduction. "
                    "Please set use_segment_norm=False when reduction_strategy='segment_only'."
                )
        elif self.reduction_strategy == "period_only":
            # period_only策略必须使用全局归一化，不能使用period_norm
            if self.use_period_norm:
                raise ValueError(
                    "period_only strategy is not compatible with period_norm. "
                    "Period_only requires global normalization for proper functioning. "
                    "Please set use_period_norm=False when reduction_strategy='period_only'."
                )
            # period_only策略可以使用segment_norm
        elif self.use_period_reduction and self.use_period_norm:
            # 其他使用period_reduction的策略不兼容period_norm
            raise ValueError(
                "period_norm is not compatible with period_reduction in sequential/parallel strategies. "
                "When using period_reduction, the period dimension is transformed "
                "through basis functions, making period-wise normalization invalid. "
                "Please set use_period_norm=False when use_period_reduction=True."
            )
        
        # segment_norm和period_norm不能同时使用
        if self.use_segment_norm and self.use_period_norm:
            raise ValueError(
                "segment_norm and period_norm cannot be used together. "
                "Please choose one normalization strategy: either use_segment_norm=True "
                "or use_period_norm=True, but not both."
            )

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        if self.seq_len > self.seg_num_x * self.period_len:
            self.pad_seq_len = (self.seg_num_x + 1) * self.period_len - self.seq_len
            self.seg_num_x += 1
        if self.pred_len > self.seg_num_y * self.period_len:
            self.seg_num_y += 1

        self.individual = configs.individual

        # 根据降维开关设置线性层
        self._setup_linear_layers()

    def _setup_linear_layers(self):
        """设置线性层结构，支持四种降维策略"""
        if self.individual:
            # Individual模式：为每个变量创建独立的线性层
            # Segment降维层（当segment reduction启用时）
            if self.use_segment_reduction:
                self.segment_ts2basis = nn.ModuleList()
                self.segment_basis2ts = nn.ModuleList()
                for i in range(self.enc_in):
                    self.segment_ts2basis.append(
                        nn.Linear(self.seg_num_x, self.basis_num)
                    )
                    self.segment_basis2ts.append(
                        nn.Linear(self.basis_num, self.seg_num_y)
                    )

            # Period降维层（当period reduction启用时）- 基于延迟嵌入
            if self.use_period_reduction:
                self.period_ts2basis = nn.ModuleList()
                self.period_basis2ts = nn.ModuleList()
                for i in range(self.enc_in):
                    # 延迟嵌入降维：period_len个延迟嵌入 -> period_basis_num个系数
                    self.period_ts2basis.append(
                        nn.Linear(self.period_len, self.period_basis_num)
                    )
                    # 重建：period_basis_num个系数 -> period_len个延迟嵌入
                    self.period_basis2ts.append(
                        nn.Linear(self.period_basis_num, self.period_len)
                    )
                    
                # 特殊情况处理
                if self.reduction_strategy == "parallel":
                    # 并行策略需要额外的映射层：seg_num_x -> seg_num_y
                    self.period_seg_mapping = nn.ModuleList()
                    for i in range(self.enc_in):
                        self.period_seg_mapping.append(
                            nn.Linear(self.seg_num_x, self.seg_num_y)
                        )
                elif self.reduction_strategy == "period_only":
                    # period_only策略需要从seg_num_x直接映射到seg_num_y
                    self.period_output_mapping = nn.ModuleList()
                    for i in range(self.enc_in):
                        self.period_output_mapping.append(
                            nn.Linear(self.seg_num_x, self.seg_num_y)
                        )
        else:
            # Shared模式：所有变量共享线性层
            # Segment降维层（当segment reduction启用时）
            if self.use_segment_reduction:
                self.segment_ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
                self.segment_basis2ts = nn.Linear(self.basis_num, self.seg_num_y)

            # Period降维层（当period reduction启用时）- 基于延迟嵌入
            if self.use_period_reduction:
                # 延迟嵌入降维：period_len个延迟嵌入 -> period_basis_num个系数
                self.period_ts2basis = nn.Linear(self.period_len, self.period_basis_num)
                # 重建：period_basis_num个系数 -> period_len个延迟嵌入
                self.period_basis2ts = nn.Linear(self.period_basis_num, self.period_len)
                
                # 特殊情况处理
                if self.reduction_strategy == "parallel":
                    # 并行策略需要额外的映射层：seg_num_x -> seg_num_y
                    self.period_seg_mapping = nn.Linear(self.seg_num_x, self.seg_num_y)
                elif self.reduction_strategy == "period_only":
                    # period_only策略需要从seg_num_x直接映射到seg_num_y
                    self.period_output_mapping = nn.Linear(self.seg_num_x, self.seg_num_y)

    def _apply_period_reduction(self, x, b, c):
        """
        应用Period维度降维 - 基于延迟嵌入冗余性
        
        输入: x (b, c, period_len, seg_num_x)
        输出: x_transformed (b, c, period_len, seg_num_x), period_basis_combined
        
        Period降维设计：
        - 对于每个segment位置，构建其跨period的延迟嵌入 [period0, period1, ..., period23]
        - 有period_len个这样的延迟嵌入，每个长度为seg_num_x
        - 用period_basis_num个基函数（每个长度为seg_num_x）来表示这些延迟嵌入
        - 目标：在period维度上找到冗余性，用更少的基表示period模式
        """
        if not self.use_period_reduction:
            return x, None
            
        period_basis_combined = None
        
        if self.individual:
            # Individual模式：每个变量使用独立的变换
            x_period_basis_list = []
            x_output = torch.zeros_like(x)  # 保持相同形状 (b, c, period_len, seg_num_x)
            
            for i in range(self.enc_in):
                # period_len个延迟嵌入，每个长度为seg_num_x
                # x[:, i, :, :] 形状: (b, period_len, seg_num_x)
                period_data = x[:, i, :, :]  # (b, period_len, seg_num_x)
                
                # 获取period basis系数：对period_len个延迟嵌入进行降维表示
                period_data_transposed = period_data.permute(0, 2, 1)  # (b, seg_num_x, period_len)
                period_data_flat = period_data_transposed.reshape(-1, self.period_len)  # (b*seg_num_x, period_len)
                
                # 获取系数
                period_coeffs = self.period_ts2basis[i](period_data_flat)  # (b*seg_num_x, period_basis_num)
                period_basis = period_coeffs.reshape(b, self.seg_num_x, self.period_basis_num)
                
                # 重建
                period_reconstructed = self.period_basis2ts[i](period_coeffs)  # (b*seg_num_x, period_len)
                period_pred_transposed = period_reconstructed.reshape(b, self.seg_num_x, self.period_len)
                period_pred = period_pred_transposed.permute(0, 2, 1)  # (b, period_len, seg_num_x)
                
                x_output[:, i, :, :] = period_pred
                x_period_basis_list.append(period_basis)
            
            # 收集period basis
            period_basis_combined = torch.stack(x_period_basis_list, dim=1)  # (b, c, seg_num_x, period_basis_num)
            period_basis_combined = period_basis_combined.reshape(-1, self.seg_num_x, self.period_basis_num)
            x = x_output
        else:
            # Shared模式：所有变量共享延迟嵌入变换
            # x形状: (b, c, period_len, seg_num_x)
            # 转置以便在period维度上降维
            x_transposed = x.permute(0, 1, 3, 2)  # (b, c, seg_num_x, period_len)
            x_flat = x_transposed.reshape(-1, self.period_len)  # (b*c*seg_num_x, period_len)
            
            # 获取period basis系数
            period_coeffs = self.period_ts2basis(x_flat)  # (b*c*seg_num_x, period_basis_num)
            period_basis = period_coeffs.reshape(b, c, self.seg_num_x, self.period_basis_num)
            
            # 重建
            period_reconstructed = self.period_basis2ts(period_coeffs)  # (b*c*seg_num_x, period_len)
            x_reconstructed = period_reconstructed.reshape(b, c, self.seg_num_x, self.period_len)
            period_pred = x_reconstructed.permute(0, 1, 3, 2)  # (b, c, period_len, seg_num_x)
            
            x = period_pred
            
            # 收集period basis - 调整为与segment basis一致的结构
            period_basis_combined = period_basis.reshape(-1, self.seg_num_x, self.period_basis_num)
        
        return x, period_basis_combined

    def _apply_segment_reduction(self, x, b, c):
        """
        应用Segment维度降维
        
        输入: x (b, c, period_len, seg_num_x)
        输出: x_transformed (b, c, period_len, seg_num_y), segment_basis_combined
        
        Segment降维：发现相似segment间的冗余性，学习跨周期的时序模式
        """
        x_segment_basis_list = []
        x_output = torch.zeros([b, c, self.period_len, self.seg_num_y], 
                             dtype=x.dtype, device=x.device)
        
        for i in range(self.enc_in):
            # x[:, i, :, :] 形状: (b, period_len, seg_num_x)
            if self.individual:
                segment_basis = self.segment_ts2basis[i](x[:, i, :, :])  # (b, period_len, basis_num)
                segment_pred = self.segment_basis2ts[i](segment_basis)   # (b, period_len, seg_num_y)
            else:
                segment_basis = self.segment_ts2basis(x[:, i, :, :])     # (b, period_len, basis_num)
                segment_pred = self.segment_basis2ts(segment_basis)      # (b, period_len, seg_num_y)
            
            x_output[:, i, :, :] = segment_pred
            x_segment_basis_list.append(segment_basis)
        
        x = x_output  # 更新x: (b, c, period_len, seg_num_y)
        
        # 收集segment basis
        segment_basis_combined = torch.stack(x_segment_basis_list, dim=1)  # (b, c, period_len, basis_num)
        segment_basis_combined = segment_basis_combined.reshape(-1, self.period_len, self.basis_num)
        
        return x, segment_basis_combined

    def _apply_period_only_reduction(self, x, b, c):
        """
        应用仅Period降维策略
        
        输入: x (b, c, period_len, seg_num_x)
        输出: x_transformed (b, c, period_len, seg_num_y), None, period_basis_combined
        
        Period Only策略设计：
        1. 仅使用Period降维，不使用Segment降维
        2. Period降维后直接映射到目标segment数量
        3. 适用于周期性很强但segment间相似度不高的数据
        """
        # Step 1: Period降维 - 基于延迟嵌入冗余性
        x_period, period_basis_combined = self._apply_period_reduction(x, b, c)
        
        # Step 2: 直接映射到目标segment数量 (seg_num_x -> seg_num_y)
        x_output_list = []
        for i in range(self.enc_in):
            # x_period[:, i, :, :] 形状: (b, period_len, seg_num_x)
            period_data = x_period[:, i, :, :]  # (b, period_len, seg_num_x)
            
            if self.individual:
                # Individual模式：每个变量使用独立的映射
                period_mapped = self.period_output_mapping[i](period_data)  # (b, period_len, seg_num_y)
            else:
                # Shared模式：所有变量共享映射
                period_mapped = self.period_output_mapping(period_data)     # (b, period_len, seg_num_y)
            
            x_output_list.append(period_mapped)
        
        # 合并所有变量的结果
        x_output = torch.stack(x_output_list, dim=1)  # (b, c, period_len, seg_num_y)
        
        return x_output, None, period_basis_combined  # segment_basis为None

    def _apply_parallel_reduction(self, x, b, c):
        """
        应用并行降维策略 - Period和Segment降维并行执行，每个分支独立归一化
        
        输入: x (b, c, period_len, seg_num_x)
        输出: x_transformed (b, c, period_len, seg_num_y), segment_basis_combined, period_basis_combined
        
        并行策略设计：
        1. 为两个分支分别进行独立的归一化
        2. Period分支：Period降维 + 映射 + Period逆归一化
        3. Segment分支：Segment降维 + Segment逆归一化
        4. 融合两个分支的结果
        """
        # 为两个分支创建独立的数据副本
        x_period_branch = x.clone()
        x_segment_branch = x.clone()
        
        # Period分支处理
        # Step 1a: Period分支独立归一化（仅全局归一化）
        x_period_flat = x_period_branch.reshape(-1, self.period_len, self.seg_num_x)
        x_period_norm, period_norm_stats = self._normalize_for_period_branch(x_period_flat, b, c)
        x_period_branch = x_period_norm.reshape(b, c, self.period_len, self.seg_num_x)
        
        # Step 1b: Period降维
        x_period, period_basis_combined = self._apply_period_reduction(x_period_branch, b, c)
        
        # Step 1c: Period结果映射到seg_num_y
        x_period_flat = x_period.reshape(-1, self.seg_num_x)  # (b*c*period_len, seg_num_x)
        if self.individual:
            x_period_mapped_list = []
            for i in range(self.enc_in):
                period_data = x_period[:, i, :, :].reshape(-1, self.seg_num_x)  # (b*period_len, seg_num_x)
                period_mapped = self.period_seg_mapping[i](period_data)  # (b*period_len, seg_num_y)
                period_mapped = period_mapped.reshape(b, self.period_len, self.seg_num_y)
                x_period_mapped_list.append(period_mapped)
            x_period_mapped = torch.stack(x_period_mapped_list, dim=1)  # (b, c, period_len, seg_num_y)
        else:
            period_mapped_flat = self.period_seg_mapping(x_period_flat)  # (b*c*period_len, seg_num_y)
            x_period_mapped = period_mapped_flat.reshape(b, c, self.period_len, self.seg_num_y)
        
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

    def _apply_basis_transformation(self, x, b, c):
        """
        应用basis变换 - 支持四种降维策略
        输入: x (bc, period_len, seg_num_x)
        输出: x_transformed (bc, period_len, seg_num_y), segment_basis, period_basis
        
        支持的策略：
        1. segment_only: 仅Segment降维
        2. period_only: 仅Period降维
        3. sequential: 串行降维（先Period，再Segment）
        4. parallel: 并行降维（Period和Segment并行，然后融合）
        """
        # x的形状: (bc, period_len, seg_num_x) -> (b, c, period_len, seg_num_x)
        x = x.reshape(b, c, self.period_len, self.seg_num_x)
        
        if self.reduction_strategy == "segment_only":
            # 策略1: 仅Segment降维
            x, segment_basis_combined = self._apply_segment_reduction(x, b, c)
            period_basis_combined = None
            
        elif self.reduction_strategy == "period_only":
            # 策略2: 仅Period降维
            x, segment_basis_combined, period_basis_combined = self._apply_period_only_reduction(x, b, c)
            # segment_basis_combined is None for period_only
            
        elif self.reduction_strategy == "sequential":
            # 策略3: 串行降维（先Period，再Segment）
            # Step 1: Period维度降维 - 基于延迟嵌入冗余性
            x, period_basis_combined = self._apply_period_reduction(x, b, c)
            # Step 2: Segment维度降维
            x, segment_basis_combined = self._apply_segment_reduction(x, b, c)
            
        elif self.reduction_strategy == "parallel":
            # 策略4: 并行降维（Period和Segment并行，然后融合）
            x, segment_basis_combined, period_basis_combined = self._apply_parallel_reduction(x, b, c)
            
        else:
            raise ValueError(f"Unsupported reduction strategy: {self.reduction_strategy}")

        # 最终输出: 保证 (bc, period_len, seg_num_y) 的形状
        x = x.reshape(-1, self.period_len, self.seg_num_y)
        
        return x, segment_basis_combined, period_basis_combined

    def _normalize_input(self, x, b, c):
        """
        input: x (bc, p, n)
        output: x_normalized, norm_stats (dict)
        
        支持三种归一化策略：
        1. period_norm: 在period维度上归一化
        2. segment_norm: 在segment维度上归一化
        3. global_norm: 全局归一化（默认）
        """
        if self.use_period_norm:
            # Period归一化：对每个period内的所有segment求均值
            period_mean = torch.mean(x, dim=-1, keepdim=True)  # (bc, period_len, 1)
            x = x - period_mean
            return x, {"period_mean": period_mean}
        elif self.use_segment_norm:
            # Segment归一化：对每个segment内的所有period求均值
            # x: (bc, period_len, seg_num_x) -> (b, c, period_len, seg_num_x)
            x_reshaped = x.reshape(b, c, self.period_len, self.seg_num_x)
            segment_mean = torch.mean(x_reshaped, dim=2, keepdim=True)  # (b, c, 1, seg_num_x)
            x_reshaped = x_reshaped - segment_mean
            x = x_reshaped.reshape(-1, self.period_len, self.seg_num_x)
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
        input: x (bc, period_len, seg_num_y) - 形状固定
        output: x_denorm
        
        支持三种逆归一化策略：
        1. period_norm: 加回period_mean
        2. segment_norm: 加回segment_mean（需要处理segment数量变化）
        3. global_norm: 加回全局mean
        """
        if self.use_period_norm:
            # Period归一化：直接加回period_mean
            # period_mean形状: (bc, period_len, 1)，需要广播到(bc, period_len, seg_num_y)
            period_mean = norm_stats["period_mean"]
            x = x + period_mean
        elif self.use_segment_norm:
            # Segment归一化：需要处理segment数量从seg_num_x到seg_num_y的变化
            # segment_mean形状: (b, c, 1, seg_num_x)
            segment_mean = norm_stats["segment_mean"]  # (b, c, 1, seg_num_x)
            
            if self.seg_num_x == self.seg_num_y:
                # 如果segment数量不变，直接使用原有的mean
                x_reshaped = x.reshape(b, c, self.period_len, self.seg_num_y)
                x_reshaped = x_reshaped + segment_mean
                x = x_reshaped.reshape(-1, self.period_len, self.seg_num_y)
            else:
                # 如果segment数量改变，需要调整segment_mean的大小
                # 方法1: 插值到新的segment数量
                segment_mean_expanded = F.interpolate(
                    segment_mean.squeeze(2),  # (b, c, seg_num_x)
                    size=self.seg_num_y,
                    mode='linear',
                    align_corners=False
                ).unsqueeze(2)  # (b, c, 1, seg_num_y)
                
                x_reshaped = x.reshape(b, c, self.period_len, self.seg_num_y)
                x_reshaped = x_reshaped + segment_mean_expanded
                x = x_reshaped.reshape(-1, self.period_len, self.seg_num_y)
        else:
            # 全局归一化：reshape，加mean，再reshape回来
            x = x.reshape(b, c, -1)
            x = x + norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        return x

    def _normalize_for_period_branch(self, x, b, c):
        """
        Period分支的独立归一化
        仅使用全局归一化（period降维与其他归一化不兼容）
        """
        x = x.reshape(b, c, -1)
        mean = torch.mean(x, dim=-1, keepdim=True)  # (b, c, 1)
        x = x - mean
        x = x.reshape(-1, self.period_len, self.seg_num_x)
        return x, {"mean": mean}

    def _normalize_for_segment_branch(self, x, b, c):
        """
        Segment分支的独立归一化
        根据配置使用segment_norm或全局归一化
        """
        if self.use_segment_norm:
            # Segment归一化
            x_reshaped = x.reshape(b, c, self.period_len, self.seg_num_x)
            segment_mean = torch.mean(x_reshaped, dim=2, keepdim=True)  # (b, c, 1, seg_num_x)
            x_reshaped = x_reshaped - segment_mean
            x = x_reshaped.reshape(-1, self.period_len, self.seg_num_x)
            return x, {"segment_mean": segment_mean}
        else:
            # 全局归一化
            x = x.reshape(b, c, -1)
            mean = torch.mean(x, dim=-1, keepdim=True)  # (b, c, 1)
            x = x - mean
            x = x.reshape(-1, self.period_len, self.seg_num_x)
            return x, {"mean": mean}

    def _denormalize_for_period_branch(self, x, norm_stats, b, c):
        """
        Period分支的独立逆归一化
        """
        x = x.reshape(b, c, -1)
        x = x + norm_stats["mean"]  # (b, c, 1)
        x = x.reshape(-1, self.period_len, self.seg_num_x)
        return x

    def _denormalize_for_segment_branch(self, x, norm_stats, b, c):
        """
        Segment分支的独立逆归一化
        """
        if "segment_mean" in norm_stats:
            # Segment归一化的逆操作
            segment_mean = norm_stats["segment_mean"]  # (b, c, 1, seg_num_x)
            if self.seg_num_x == self.seg_num_y:
                x_reshaped = x.reshape(b, c, self.period_len, self.seg_num_y)
                x_reshaped = x_reshaped + segment_mean
                x = x_reshaped.reshape(-1, self.period_len, self.seg_num_y)
            else:
                # 插值处理segment数量变化
                segment_mean_expanded = F.interpolate(
                    segment_mean.squeeze(2),  # (b, c, seg_num_x)
                    size=self.seg_num_y,
                    mode='linear',
                    align_corners=False
                ).unsqueeze(2)  # (b, c, 1, seg_num_y)
                
                x_reshaped = x.reshape(b, c, self.period_len, self.seg_num_y)
                x_reshaped = x_reshaped + segment_mean_expanded
                x = x_reshaped.reshape(-1, self.period_len, self.seg_num_y)
        else:
            # 全局归一化的逆操作
            x = x.reshape(b, c, -1)
            x = x + norm_stats["mean"]  # (b, c, 1)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        return x

    def forward(
        self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs
    ):
        """
        x_enc: b t c (batch_size, seq_len, enc_in)
        out: b t c (batch_size, pred_len, enc_in)
        """
        # Use x_enc as the main input, ignore other parameters for this model
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

        # 新的Basis transformation支持两个维度的独立降维
        x, segment_basis, period_basis = self._apply_basis_transformation(x, b, c)

        # Denormalize
        x = self._denormalize_output(x, norm_stats, b, c)

        # Reshape back - 现在形状是固定的 (bc, period_len, seg_num_y)
        x = x.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y)
        x = x.permute(0, 1, 3, 2)  # (batch_size, enc_in, seg_num_y, period_len)
        x = x.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)  # b t c

        # Forward方法只返回预测结果，不计算损失
        output = x[:, : self.pred_len, :]

        # 如果需要正交损失，返回两个basis用于损失计算
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

        # 调用forward方法
        forward_output = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        # 处理forward方法的返回值
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

        # 在training_step中使用分层权重分别计算两个正交损失
        if self.use_orthogonal:
            total_orthogonal_loss = 0
            weighted_segment_loss = 0
            weighted_period_loss = 0
            
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
            
            # 总损失
            total_loss = loss + total_orthogonal_loss

            # 记录各项损失和权重
            self.log("train_loss_main", loss, on_epoch=True)
            self.log("train_loss_orthogonal_total", total_orthogonal_loss, on_epoch=True)
            self.log("train_loss", total_loss, on_epoch=True)
            self.log("segment_ortho_weight", self.segment_ortho_weight, on_epoch=True)
            self.log("period_ortho_weight", self.period_ortho_weight, on_epoch=True)
            
            # 记录损失占比
            main_ratio = loss / total_loss * 100
            ortho_ratio = total_orthogonal_loss / total_loss * 100
            self.log("train_main_loss_ratio", main_ratio, on_epoch=True)
            self.log("train_ortho_loss_ratio", ortho_ratio, on_epoch=True)

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

        # 调用forward方法
        forward_output = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        # 处理forward方法的返回值，验证时只需要预测结果
        if self.use_orthogonal:
            outputs, _, _ = forward_output  # 忽略两个basis
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

        # 调用forward方法
        forward_output = self(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
        )

        # 处理forward方法的返回值，测试时只需要预测结果
        if self.use_orthogonal:
            outputs, _, _ = forward_output  # 忽略两个basis
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
