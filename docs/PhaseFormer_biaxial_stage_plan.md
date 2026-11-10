# 双轴 PhaseFormer 阶段实验计划

起始日期：2026-11-10。起点：代码仓库提交 6745a6c，论文仓库提交 1480799。

## 实现

- 代码：`scripts/biaxial/`（`model.py` 模型，`train.py` 训练与评估，`worker.py` 单 GPU 任务执行，`launch.py` 多进程启动，`configs.py` 任务表，`aggregate.py` 汇总与判定）。
- 服务器：`~/niuyiming/PhaseFormer_biaxial_20261110`（数据集链接到 `~/niuyiming/PhaseFormer/resources`），环境 conda `time`（torch 2.6），GPU 4、5。
- 协议：输入长度 720，Adam 学习率 1e-3，Huber 损失，最多 30 个 epoch，patience 8，批大小 256，使用仓库自带数据集类（标准化数据，70/10/20 或 ETT 12/4/4 个月划分）。最优验证 Huber 损失对应的参数用于报告验证与测试指标。
- M0 为本阶段重新实现的 PhaseFormer+（相位路径、趋势路径、双门控），在同一协议下重新训练，数值与论文主表不直接比较。基础容量：3 层、d=4、8 个 Router、1 个头、dropout 0.1、门控初值 0.2、周期 24（ETTm1 取 48）。

## 筛选规则

- 筛选集：ETTh1、ETTh2、ETTm1、Weather，预测长度 96 和 720，种子 2021。
- 选择只看验证集 MSE。新配置的平均相对下降不少于 1% 且在 4 个数据集中至少 3 个不劣于对照，才算优于对照。测试指标仅记录。

## 实验组

E0 基线与容量对照（M0，M0W-P 参数量匹配，M0W-F 计算量匹配）；E1 双轴输入表示与 Router 布局（B0 仅相位 Router，B1 仅周期 Router，B2 独立双分支，B3 交替双轴）；E2 层数与混合布局；E3 低维表示扫描；E4 解码器与锚点；E5、E6 诊断与误差分解；E7 最终评测。详细内容见对话中的方案。
