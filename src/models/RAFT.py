import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from src.models.pl_bases.default_module import DefaultPLModule


class RetrievalTool:
    """
    检索工具类，用于从训练数据中检索相似的时间序列模式
    基于多尺度分解和相关性计算来找到最相似的序列
    """
    def __init__(
        self,
        seq_len,        # 输入序列长度
        pred_len,       # 预测序列长度
        channels,       # 特征通道数
        n_period=3,     # 多尺度分解的周期数量
        temperature=0.1, # softmax温度参数，控制检索概率分布
        topm=20,        # 检索时返回的最相似序列数量
        with_dec=False, # 是否包含解码器
        return_key=False, # 是否返回检索键
        device=None,    # 进行预处理计算的设备（如 'cuda:0' 或 'cpu'）
    ):
        # 定义多尺度分解的周期数，从16到1的不同粒度
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period :]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.n_period = n_period
        # 按降序排列周期数，确保从粗粒度到细粒度
        self.period_num = sorted(period_num, reverse=True)

        self.temperature = temperature
        self.topm = topm

        self.with_dec = with_dec
        self.return_key = return_key
        # 预处理和分解所用设备（默认CPU）
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

    def prepare_dataset(self, train_data, device=None):
        """
        准备训练数据集，将数据转换为适合检索的格式
        
        Args:
            train_data: 训练数据，包含输入序列和标签
        """
        # 选择计算设备
        device = (
            torch.device(device) if isinstance(device, str) else device
        ) if device is not None else self.device

        train_data_all = []
        y_data_all = []

        print("preparing train data")

        for i in range(len(train_data)):
            td = train_data[i]
            # 适配本项目数据集返回: (seq_x, seq_y, seq_x_mark, seq_y_mark)
            # 训练相似度使用历史窗口 seq_x
            train_data_all.append(td[0])

            # 检索目标使用未来序列 seq_y 的尾部 pred_len 部分
            if self.with_dec:
                # 包含解码标签长度时，使用 label_len + pred_len
                y_data_all.append(
                    td[1][-(train_data.pred_len + train_data.label_len) :]
                )
            else:
                # 否则仅使用预测区间
                y_data_all.append(td[1][-train_data.pred_len :])

        # 将所有训练数据堆叠成张量（放置到指定设备）
        self.train_data_all = torch.as_tensor(
            np.stack(train_data_all, axis=0), device=device
        ).float()
        # 对训练数据进行多尺度分解（在指定设备上）
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all, device=device)

        # 将所有标签数据堆叠成张量（放置到指定设备）
        self.y_data_all = torch.as_tensor(
            np.stack(y_data_all, axis=0), device=device
        ).float()
        # 对标签数据进行多尺度分解（在指定设备上）
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all, device=device)

        self.n_train = self.train_data_all.shape[0]

    def decompose_mg(self, data_all, remove_offset=True, device=None):
        """
        多尺度分解函数，将时间序列分解为不同粒度的表示
        
        Args:
            data_all: 输入数据，形状为 (T, S, C)
            remove_offset: 是否移除偏移量
            
        Returns:
            mg: 多尺度分解结果，形状为 (G, T, S, C)
            offset: 偏移量，形状为 (G, T, 1, C)
        """
        from tqdm import tqdm

        # 选择计算设备
        device = (
            torch.device(device) if isinstance(device, str) else device
        ) if device is not None else self.device

        data_all = copy.deepcopy(data_all).to(device)  # T, S, C

        mg = []
        for g in tqdm(self.period_num, desc="Scales", leave=False):
            # 使用unfold操作按周期g进行滑动窗口平均
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            # 将结果重复g次以匹配原始序列长度
            cur = cur.repeat_interleave(repeats=g, dim=1)

            mg.append(cur)
        #             data_all = data_all - cur

        # 将所有尺度的结果堆叠
        mg = torch.stack(mg, dim=0).to(self.device)  # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(tqdm(mg, desc="Offsets", leave=False)):
                # 计算每个尺度的偏移量（使用最后一个时间步）
                cur_offset = data_p[:, -1:, :]
                # 从数据中减去偏移量
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None

        offset = torch.stack(offset, dim=0).to(self.device)

        return mg, offset

    def periodic_batch_corr(self, data_all, key, in_bsz=512):
        """
        计算周期性批量相关性，用于找到最相似的序列
        
        Args:
            data_all: 训练数据，形状为 (G, T, S*C)
            key: 查询键，形状为 (G, B, S*C)
            in_bsz: 内部批处理大小
            
        Returns:
            sim: 相似度矩阵，形状为 (G, B, T)
        """
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        # 计算查询键的均值并中心化
        bx = key - torch.mean(key, dim=2, keepdim=True)

        # 分批处理以节省内存
        iters = math.ceil(train_len / in_bsz)

        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            # 获取当前批次的数据
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            # 中心化当前批次数据
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)

            # 计算余弦相似度
            cur_sim = torch.bmm(
                F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2)
            )
            sim.append(cur_sim)

        # 将所有批次的相似度结果连接
        sim = torch.cat(sim, dim=2)

        return sim

    def retrieve(self, x, index, train=True):
        """
        检索函数，从训练数据中找到最相似的序列
        
        Args:
            x: 输入序列，形状为 (B, S, C)
            index: 序列索引
            train: 是否为训练模式
            
        Returns:
            pred_from_retrieval: 基于检索的预测结果，形状为 (G, B, P, C)
        """
        index = index.to(x.device)

        bsz, seq_len, channels = x.shape
        assert (seq_len == self.seq_len, channels == self.channels)

        # 对输入序列进行多尺度分解（在与 x 相同的设备上）
        x_mg, mg_offset = self.decompose_mg(x, device=x.device)  # G, B, S, C
        # NAN检测：分解后的输入
        if torch.isnan(x_mg).any():
            print("[RetrievalTool.retrieve] NAN detected in x_mg")
            print(f"  x_mg shape={x_mg.shape}, stats: min={torch.nanmin(x_mg):.6f}, max={torch.nanmax(x_mg):.6f}")
        if mg_offset is not None and torch.isnan(mg_offset).any():
            print("[RetrievalTool.retrieve] NAN detected in mg_offset")
            print(f"  mg_offset shape={mg_offset.shape}, stats: min={torch.nanmin(mg_offset):.6f}, max={torch.nanmax(mg_offset):.6f}")

        # 计算与训练数据的相似度
        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2),  # G, T, S * C
            x_mg.flatten(start_dim=2),  # G, B, S * C
        )  # G, B, T

        # NAN检测：相似度
        if torch.isnan(sim).any():
            print("[RetrievalTool.retrieve] NAN detected in similarity matrix sim before masking")
            print(f"  sim shape={sim.shape}")

        if train:
            # 训练时创建滑动窗口索引，避免数据泄露
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(
                x.device
            )
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (
                index - self.seq_len - self.pred_len + 1
            ).unsqueeze(dim=1)

            # 确保索引在有效范围内
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(
                sliding_index < self.n_train, sliding_index, self.n_train - 1
            )

            # 创建掩码，将滑动窗口内的相似度设为负无穷
            self_mask = torch.zeros((bsz, self.n_train)).to(sim.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.0)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)

            sim = sim.masked_fill_(self_mask.bool(), float("-inf"))  # G, B, T
            # 检查是否存在整行被掩蔽为 -inf 的情况
            all_neg_inf = torch.isinf(sim) & (sim < 0)
            all_neg_inf_rows = all_neg_inf.all(dim=2)
            if all_neg_inf_rows.any():
                num_bad = int(all_neg_inf_rows.sum())
                print(f"[RetrievalTool.retrieve] WARNING: {num_bad} (G,B) rows fully masked to -inf.\n  This will cause softmax to produce NaN.")
                # 打印示例位置
                bad_indices = torch.nonzero(all_neg_inf_rows, as_tuple=False)
                print(f"  Example bad rows (up to 5): {bad_indices[:5].tolist()}")

        # 重塑相似度矩阵以便进行top-k选择
        sim = sim.reshape(self.n_period * bsz, self.n_train)  # G X B, T

        # 选择top-m最相似的序列
        topm_index = torch.topk(sim, self.topm, dim=1).indices
        ranking_sim = torch.ones_like(sim) * float("-inf")

        # 只保留top-m的相似度值，其他设为负无穷
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]

        # 重塑回原始形状
        sim = sim.reshape(self.n_period, bsz, self.n_train)  # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train)  # G, B, T

        data_len, seq_len, channels = self.train_data_all.shape

        # 使用softmax计算检索概率
        if self.temperature is None or self.temperature <= 0:
            print(f"[RetrievalTool.retrieve] ERROR: temperature={self.temperature} <= 0; softmax will be ill-defined.")
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu()  # G, B, T

        # NAN检测：softmax后的概率
        if torch.isnan(ranking_prob).any():
            print("[RetrievalTool.retrieve] NAN detected in ranking_prob after softmax")
            # 检测是否由 all -inf 行导致
            is_neg_inf = torch.isinf(ranking_sim) & (ranking_sim < 0)
            all_neg_inf_rows = is_neg_inf.all(dim=2)
            if all_neg_inf_rows.any():
                print("  Cause: softmax over all -inf due to full masking or invalid similarities.")
            # 打印数值范围
            finite_mask = torch.isfinite(ranking_sim)
            if finite_mask.any():
                print(f"  ranking_sim finite stats: min={ranking_sim[finite_mask].min():.6f}, max={ranking_sim[finite_mask].max():.6f}")

        # 获取标签数据的多尺度表示
        y_data_all = self.y_data_all_mg.flatten(start_dim=2)  # G, T, P * C
        if torch.isnan(y_data_all).any():
            print("[RetrievalTool.retrieve] WARNING: NaN found in y_data_all_mg (labels). This will propagate to outputs.")

        # 基于检索概率和标签数据计算预测结果
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(
            self.n_period, bsz, -1, channels
        )
        pred_from_retrieval = pred_from_retrieval.to(x.device)

        if torch.isnan(pred_from_retrieval).any():
            print("[RetrievalTool.retrieve] NAN detected in pred_from_retrieval after bmm")
            # 打印一部分行的概率和
            prob_sum = ranking_prob.sum(dim=2)
            print(f"  ranking_prob row-sum stats: min={prob_sum.min():.6f}, max={prob_sum.max():.6f}, mean={prob_sum.mean():.6f}")

        return pred_from_retrieval

    def retrieve_all(self, data, train=False, device=torch.device("cpu")):
        """
        对所有数据进行检索
        
        Args:
            data: 输入数据
            train: 是否为训练模式
            device: 计算设备
            
        Returns:
            retrievals: 所有检索结果的连接
        """
        assert self.train_data_all_mg != None

        # 创建数据加载器
        rt_loader = DataLoader(
            data, batch_size=1024, shuffle=False, num_workers=8, drop_last=False
        )

        retrievals = []
        cur_ptr = 0
        with torch.no_grad():
            for batch in tqdm(rt_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                bsz = batch_x.shape[0]
                index = torch.arange(cur_ptr, cur_ptr + bsz)
                cur_ptr += bsz
                # 对每个批次进行检索
                pred_from_retrieval = self.retrieve(
                    batch_x.float().to(device), index, train=train
                )
                pred_from_retrieval = pred_from_retrieval.cpu()
                # NAN检测：当前批次检索结果
                if torch.isnan(pred_from_retrieval).any():
                    print(f"[RetrievalTool.retrieve_all] NAN detected in batch retrievals. train={train}")
                    print(f"  batch index range: [{int(index[0])}, {int(index[-1])}] size={bsz}")
                    # 打印每个尺度的占比
                    nan_ratio_per_g = torch.isnan(pred_from_retrieval).flatten(2).float().mean(dim=2)
                    print(f"  NaN ratio per scale (first 5 batches if many): {nan_ratio_per_g[:, :min(5, nan_ratio_per_g.shape[1])].tolist()}")
                retrievals.append(pred_from_retrieval)

        # 连接所有检索结果
        retrievals = torch.cat(retrievals, dim=1)

        return retrievals


class RAFT(DefaultPLModule):
    """
    RAFT模型主类 - 基于检索增强的时间序列预测模型
    论文链接: https://arxiv.org/pdf/2205.13504.pdf
    
    RAFT是一个基于检索增强的时间序列预测模型，通过从历史数据中检索相似模式
    来辅助预测，特别适用于长期预测任务
    """

    def __init__(self, configs, individual=False):
        """
        初始化RAFT模型
        
        Args:
            configs: 配置对象，包含模型的各种超参数
            individual: 布尔值，是否为不同变量使用独立的模型
        """
        super().__init__(configs)
        
        # 基础配置
        self.seq_len = configs.seq_len
        if (
            configs.task_name == "classification"
            or configs.task_name == "anomaly_detection"
            or configs.task_name == "imputation"
        ):
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
            
        self.channels = configs.enc_in
        self.task_name = configs.task_name

        # 线性层：将输入序列长度映射到预测长度
        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        # 检索相关参数
        self.n_period = getattr(configs, 'n_period', 3)
        self.topm = getattr(configs, 'topm', 20)
        self.temperature = getattr(configs, 'temperature', 0.1)

        # 初始化检索工具
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
            temperature=self.temperature,
            device="cpu"
        )

        self.period_num = self.rt.period_num[-1 * self.n_period :]

        # 为每个尺度创建预测模块
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len) for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        
        # 融合线性层：将检索预测和直接预测结果融合
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

        # 保存超参数
        self.save_hyperparameters()

    def prepare_dataset(self, train_data, valid_data, test_data, cache_dir=None, cache_tag=None, force_recompute=False):
        """
        准备数据集，包括训练、验证和测试数据的检索
        
        Args:
            train_data: 训练数据
            valid_data: 验证数据
            test_data: 测试数据
            cache_dir: 缓存目录（保存/加载预检索结果）
            cache_tag: 缓存标签（用来区分不同数据/超参组合）
            force_recompute: 是否强制重新计算并覆盖缓存
        """
        # 缓存路径准备
        cache_dir = cache_dir or "./log/training_results/RAFT_cache"
        os.makedirs(cache_dir, exist_ok=True)
        # 自动构造tag（若未提供），调用方最好提供明确的tag
        if cache_tag is None:
            cache_tag = f"S{self.seq_len}-P{self.pred_len}-C{self.channels}-np{self.n_period}-top{self.topm}"

        base = os.path.join(cache_dir, cache_tag)
        meta_path = base + "_meta.json"
        train_path = base + "_train.pt"
        valid_path = base + "_valid.pt"
        test_path = base + "_test.pt"

        can_load = (not force_recompute) and os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)

        if can_load:
            try:
                print(f"Loading RAFT retrieval cache: {base}_*.pt")
                train_rt = torch.load(train_path, map_location="cpu")
                valid_rt = torch.load(valid_path, map_location="cpu")
                test_rt = torch.load(test_path, map_location="cpu")

                # 基础校验
                if train_rt.dim() != 4 or valid_rt.dim() != 4 or test_rt.dim() != 4:
                    raise RuntimeError("Cached retrieval tensors have unexpected dims.")

                self.retrieval_dict = {
                    "train": train_rt.detach(),
                    "valid": valid_rt.detach(),
                    "test": test_rt.detach(),
                }

                # 重置epoch指针
                self._epoch_ptr = {"train": 0, "valid": 0, "test": 0}
                print("Retrieval cache loaded successfully.")
                return
            except Exception as e:
                print(f"Failed to load cache due to: {e}. Will recompute and overwrite cache.")

        # 若无法加载缓存，则执行预检索计算并落盘
        # 准备训练数据的检索
        self.rt.prepare_dataset(train_data, device=self.device)

        self.retrieval_dict = {}

        # 选择可用设备用于批量检索（与Lightning设备无关）
        compute_device = "cpu"

        print("Doing Train Retrieval")
        train_rt = self.rt.retrieve_all(train_data, train=True, device=compute_device)

        print("Doing Valid Retrieval")
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=compute_device)

        print("Doing Test Retrieval")
        test_rt = self.rt.retrieve_all(test_data, train=False, device=compute_device)

        # 将检索结果保存到CPU并落盘
        train_cpu = train_rt.detach().cpu()
        valid_cpu = valid_rt.detach().cpu()
        test_cpu = test_rt.detach().cpu()

        torch.save(train_cpu, train_path)
        torch.save(valid_cpu, valid_path)
        torch.save(test_cpu, test_path)

        meta = {
            "seq_len": int(self.seq_len),
            "pred_len": int(self.pred_len),
            "channels": int(self.channels),
            "n_period": int(self.n_period),
            "topm": int(self.topm),
            "temperature": float(self.temperature),
            "train_len": int(len(train_data)),
            "valid_len": int(len(valid_data)),
            "test_len": int(len(test_data)),
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: failed to write cache meta: {e}")

        # 删除检索工具以释放内存
        del self.rt
        torch.cuda.empty_cache()

        # 存储检索结果（CPU侧）
        self.retrieval_dict["train"] = train_cpu
        self.retrieval_dict["valid"] = valid_cpu
        self.retrieval_dict["test"] = test_cpu

        # 迭代指针（跟踪当前epoch内的样本偏移）
        self._epoch_ptr = {"train": 0, "valid": 0, "test": 0}

    def on_train_epoch_start(self):
        # 每个训练epoch开始时重置指针
        self._epoch_ptr["train"] = 0

    def on_validation_epoch_start(self):
        self._epoch_ptr["valid"] = 0

    def on_test_epoch_start(self):
        self._epoch_ptr["test"] = 0

    def encoder(self, x, index, mode):
        """
        编码器函数，主要的预测逻辑
        
        Args:
            x: 输入序列，形状为 (B, S, C)
            index: 序列索引
            mode: 模式（train/valid/test）
            
        Returns:
            pred: 预测结果，形状为 (B, P, C)
        """
        # 索引用于CPU侧索引预检索tensor，保持为CPU Long
        index = index.long().cpu()

        bsz, seq_len, channels = x.shape
        assert (seq_len == self.seq_len, channels == self.channels)
        
        # NAN检测：检查输入数据
        if torch.isnan(x).any():
            print(f"⚠️  NAN detected in input x at mode={mode}")
            print(f"   Input shape: {x.shape}")
            print(f"   Input stats: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")
            print(f"   NAN positions: {torch.isnan(x).nonzero()}")
            print(f"   Index: {index}")
            # 打印输入数据的详细信息
            for i in range(min(3, bsz)):  # 只打印前3个样本
                print(f"   Sample {i}: {x[i].flatten()[:10]}...")  # 只显示前10个值

        # 计算偏移量并标准化输入
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        
        # NAN检测：检查标准化后的输入
        if torch.isnan(x_norm).any():
            print(f"⚠️  NAN detected in x_norm at mode={mode}")
            print(f"   x_norm shape: {x_norm.shape}")
            print(f"   x_offset shape: {x_offset.shape}")
            print(f"   x_offset stats: min={x_offset.min():.6f}, max={x_offset.max():.6f}, mean={x_offset.mean():.6f}")

        # 直接预测：使用线性层将输入序列映射到预测长度
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # B, P, C
        
        # NAN检测：检查直接预测结果
        if torch.isnan(x_pred_from_x).any():
            print(f"⚠️  NAN detected in x_pred_from_x at mode={mode}")
            print(f"   x_pred_from_x shape: {x_pred_from_x.shape}")
            print(f"   x_pred_from_x stats: min={x_pred_from_x.min():.6f}, max={x_pred_from_x.max():.6f}, mean={x_pred_from_x.mean():.6f}")
            print(f"   linear_x weight stats: min={self.linear_x.weight.min():.6f}, max={self.linear_x.weight.max():.6f}")
            print(f"   linear_x bias stats: min={self.linear_x.bias.min():.6f}, max={self.linear_x.bias.max():.6f}")

        # 获取检索预测结果
        pred_from_retrieval = self.retrieval_dict[mode][:, index]  # G, B, P, C
        pred_from_retrieval = pred_from_retrieval.to(self.device)
        
        # NAN检测：检查检索结果
        # if torch.isnan(pred_from_retrieval).any():
        #     print(f"⚠️  NAN detected in pred_from_retrieval at mode={mode}")
        #     print(f"   pred_from_retrieval shape: {pred_from_retrieval.shape}")
        #     print(f"   pred_from_retrieval stats: min={pred_from_retrieval.min():.6f}, max={pred_from_retrieval.max():.6f}, mean={pred_from_retrieval.mean():.6f}")
        #     print(f"   Index used: {index}")
        #     print(f"   Retrieval dict keys: {list(self.retrieval_dict.keys())}")
        #     print(f"   Retrieval dict shapes: {[(k, v.shape) for k, v in self.retrieval_dict.items()]}")

        retrieval_pred_list = []

        # 压缩重复维度并应用预测模块
        for i, pr in enumerate(pred_from_retrieval):
            assert (bsz, self.pred_len, channels) == pr.shape
            g = self.period_num[i]
            
            # # NAN检测：检查每个尺度的检索结果
            # if torch.isnan(pr).any():
            #     print(f"⚠️  NAN detected in retrieval result at scale {i} (period={g}) at mode={mode}")
            #     print(f"   Scale {i} shape: {pr.shape}")
            #     print(f"   Scale {i} stats: min={pr.min():.6f}, max={pr.max():.6f}, mean={pr.mean():.6f}")
            
            # 重塑为 (B, P//g, g, C) 并取第一个时间步
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]

            # 应用预测模块
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)
            
            # # NAN检测：检查预测模块输出
            # if torch.isnan(pr).any():
            #     print(f"⚠️  NAN detected in retrieval_pred[{i}] output at mode={mode}")
            #     print(f"   retrieval_pred[{i}] weight stats: min={self.retrieval_pred[i].weight.min():.6f}, max={self.retrieval_pred[i].weight.max():.6f}")
            #     print(f"   retrieval_pred[{i}] bias stats: min={self.retrieval_pred[i].bias.min():.6f}, max={self.retrieval_pred[i].bias.max():.6f}")

            retrieval_pred_list.append(pr)

        # 将所有尺度的预测结果相加
        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)
        retrieval_pred_list = retrieval_pred_list.sum(dim=1)
        
        # # NAN检测：检查融合前的检索预测结果
        # if torch.isnan(retrieval_pred_list).any():
        #     print(f"⚠️  NAN detected in final retrieval_pred_list at mode={mode}")
        #     print(f"   retrieval_pred_list shape: {retrieval_pred_list.shape}")
        #     print(f"   retrieval_pred_list stats: min={retrieval_pred_list.min():.6f}, max={retrieval_pred_list.max():.6f}, mean={retrieval_pred_list.mean():.6f}")

        # 融合直接预测和检索预测
        pred = torch.cat([x_pred_from_x, retrieval_pred_list], dim=1)
        
        # # NAN检测：检查融合后的输入
        # if torch.isnan(pred).any():
        #     print(f"⚠️  NAN detected in concatenated pred input at mode={mode}")
        #     print(f"   Concatenated pred shape: {pred.shape}")
        #     print(f"   x_pred_from_x stats: min={x_pred_from_x.min():.6f}, max={x_pred_from_x.max():.6f}, mean={x_pred_from_x.mean():.6f}")
        #     print(f"   retrieval_pred_list stats: min={retrieval_pred_list.min():.6f}, max={retrieval_pred_list.max():.6f}, mean={retrieval_pred_list.mean():.6f}")
        
        pred = (
            self.linear_pred(pred.permute(0, 2, 1))
            .permute(0, 2, 1)
            .reshape(bsz, self.pred_len, self.channels)
        )
        
        # # NAN检测：检查线性融合层输出
        # if torch.isnan(pred).any():
        #     print(f"⚠️  NAN detected in linear_pred output at mode={mode}")
        #     print(f"   linear_pred weight stats: min={self.linear_pred.weight.min():.6f}, max={self.linear_pred.weight.max():.6f}")
        #     print(f"   linear_pred bias stats: min={self.linear_pred.bias.min():.6f}, max={self.linear_pred.bias.max():.6f}")

        # 添加回偏移量
        pred = pred + x_offset
        
        # # NAN检测：检查最终输出
        # if torch.isnan(pred).any():
        #     print(f"⚠️  NAN detected in final pred output at mode={mode}")
        #     print(f"   Final pred shape: {pred.shape}")
        #     print(f"   Final pred stats: min={pred.min():.6f}, max={pred.max():.6f}, mean={pred.mean():.6f}")
        #     print(f"   x_offset stats: min={x_offset.min():.6f}, max={x_offset.max():.6f}, mean={x_offset.mean():.6f}")
        #     print(f"   NAN positions in final pred: {torch.isnan(pred).nonzero()}")

        #     import sys
        #     sys.exit()

        return pred

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        RAFT模型的前向传播函数
        
        Args:
            x_enc: 编码器输入序列，形状为 (B, S, C)
            x_mark_enc: 编码器时间标记（可选）
            x_dec: 解码器输入（可选，RAFT不使用）
            x_mark_dec: 解码器时间标记（可选）
            
        Returns:
            pred: 预测结果，形状为 (B, P, C)
        """
        # 占位：前向仅用于inference时直接取valid顺序
        batch_size = x_enc.shape[0]
        index = torch.arange(0, batch_size)
        
        # 根据训练状态确定模式
        if self.training:
            mode = "train"
        else:
            mode = "valid"  # 验证和测试都使用valid模式
        
        # 调用编码器进行预测
        pred = self.encoder(x_enc, index, mode)
        
        return pred

    def training_step(self, batch, batch_idx):
        """
        训练步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            loss: 训练损失
        """
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        # 使用epoch指针对齐预检索顺序（确保DataLoader不shuffle）
        batch_size = batch_x.shape[0]
        start = int(self._epoch_ptr.get("train", 0))
        index = torch.arange(start, start + batch_size)
        self._epoch_ptr["train"] = start + batch_size

        # 前向传播
        outputs = self.encoder(batch_x, index, "train")

        # 确保输出长度正确
        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        # 如果指定了目标变量索引，只使用该变量
        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        # 计算损失
        loss = self._compute_loss(outputs, target)

        # 记录训练损失
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            loss: 验证损失
        """
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        # 使用epoch指针对齐
        batch_size = batch_x.shape[0]
        start = int(self._epoch_ptr.get("valid", 0))
        index = torch.arange(start, start + batch_size)
        self._epoch_ptr["valid"] = start + batch_size

        outputs = self.encoder(batch_x, index, "valid")

        # 确保输出长度正确
        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        # 如果指定了目标变量索引，只使用该变量
        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        # 计算损失
        loss = self._compute_loss(outputs, target)
        
        # 记录验证损失
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        测试步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            metrics: 测试指标字典
        """
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        # 使用epoch指针对齐
        batch_size = batch_x.shape[0]
        start = int(self._epoch_ptr.get("test", 0))
        index = torch.arange(start, start + batch_size)
        self._epoch_ptr["test"] = start + batch_size

        # 前向传播
        outputs = self.encoder(batch_x, index, "test")

        # 确保输出长度正确
        outputs = outputs[:, -self.pred_len :, :]
        target = batch_y[:, -self.pred_len :, :]

        # 如果指定了目标变量索引，只使用该变量
        if self.target_var_index != -1:
            target = target[:, :, self.target_var_index].unsqueeze(-1)

        # 计算测试指标
        from src.utils.metrics import metric
        m = metric(outputs.detach(), target.detach())
        
        # 记录测试指标
        self.log_dict({f"test_{k}": v for k, v in m.items()}, on_epoch=True)
        return m

    def _compute_loss(self, outputs, target):
        """
        计算损失函数
        
        Args:
            outputs: 模型输出
            target: 目标值
            
        Returns:
            loss: 计算得到的损失
        """
        criterion = self._get_criterion(self.args.training_args.loss_func)
        return criterion(outputs, target)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        """
        opt_cls = torch.optim.Adam
        optimizer = opt_cls(self.parameters(), lr=self.args.training_args.learning_rate)
        
        if self.args.training_args.lr_schedule_config.type == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.training_args.lr_schedule_config.tmax,
                eta_min=1e-8,
            )
            return [optimizer], [scheduler]
        return optimizer
