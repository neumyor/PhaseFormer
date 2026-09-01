# ETTh1/ETTm1 单 seed Golden 定向搜索

## 用户目标与评价

用户要求 ETTh1、ETTm1 在 H96、H192 都至少优于 Golden 0.5%，只按 seed=2021 判断，以尽快完成
定向搜索。阈值为：候选 MSE、MAE 必须分别不高于 Golden 的 `99.5%`。这是明确允许的
**test-set selection**，因此最终结果只能作为目标导向的选择轨迹，不能表述为盲测或泛化估计。

## 搜索与重跑策略

入口 `scripts/run_strict_t28_golden_hunt.py` 固定 strict-T28 单模型结构，仅搜索：

- cycle：ETTh1 为 24/48；ETTm1 为 24/48/96；
- trust region：off=`.02/.01/.005`、C=`.25/.10/.05`、W=`.60/.24/.12`、X=`.95/.50/.25`；
- loss：Huber/MAE；learning-rate multiplier：0.3/1/3。

每个配置在 H96、H192 依次 full-train（100%、最多 30 epoch、best-validation checkpoint）并读取一次
test，seed 固定为 2021。脚本把每项 test 与 Golden 的差和是否达到 0.5% 门槛写入紧凑 CSV。每条失败
命令自动重试最多 3 次；底层 `--resume` 保证中断后不会重复训练已完成实验，CSV 以配置 key 去重。

运行环境与持久化执行：

- 本机规定的 `py310` 环境不存在；经实际验证，等价的 conda `raft` 环境可识别 RTX 4090（PyTorch
  2.4.1 + CUDA 12.1、Lightning 2.5.6），因此本轮使用它而非无法在普通会话中识别 CUDA 的 `.venv`。
- 搜索由用户级 systemd transient service 承载。服务采用 `Restart=on-failure`、20 秒重启间隔；脚本本身
  每个子命令最多重试三次，底层 `--resume` 复用已完成 run。因而终端会话结束、单个训练子进程失败或机器的
  短暂服务波动都不会丢失已写入的选择轨迹。
- 状态只低频读取 `systemctl --user status phaseformer-strict-t28-golden-hunt` 和以下 CSV，避免轮询训练日志。

前台复现实例：

```bash
conda run --no-capture-output -n raft python scripts/run_strict_t28_golden_hunt.py --dataset ETTh1
conda run --no-capture-output -n raft python scripts/run_strict_t28_golden_hunt.py --dataset ETTm1
```

初始空间分别为 48/72 个配置（每个配置有两个 horizon）。达到两个 horizon 的双指标阈值后，停止该
数据集的后续搜索；若搜索空间耗尽仍无通过项，如实报告失败，不使用未记录的手调参数。

## 第二阶段：由首轮近失误驱动的精修

若首轮不能通过门槛，才执行 `scripts/run_strict_t28_golden_refinement.py`。它保持相同的
“完整 A2 预测为锚点 + 两个受限周期修正”拓扑、同一 seed、同一个 dataset 内跨 H96/H192 的共享配置，
不引入 horizon 路由。第二阶段的候选来自首轮的实际近失误，而非事后逐 horizon 拟合：

- **ETTh1**：X 信任区在 H96 已有 `−1.783%` MSE、`+0.365%` MAE 的近失误，因此只测试更长训练、
  相邻低学习率、极端 U 信任区、锚点损失/学习率平衡、较弱辅助损失与共享 `cycle=48`。
- **ETTm1**：W 信任区的 H96 为 `−0.091%` MSE、`−1.257%` MAE，故测试更低的相邻学习率、较长训练、
  X/U 信任区、锚点与辅助损失平衡，以及共享 `cycle=48/96`。

每个候选仍同时运行两个 horizon；仅当四项比较（H96/H192 的 MSE/MAE）全部 `≤ 99.5% × Golden`
才提前停止。结果附加写入各自的 `*_refinement_test_selection.csv`，并与首轮 CSV 一起保留，明确属于
test-set selection。脚本沿用每次三次重试及 `--resume`；它可被 systemd 服务在首轮结束后自动接续。

## 第三阶段：同一拓扑内的损失折中

第二阶段仍失败时，`scripts/run_strict_t28_golden_loss_refinement.py` 只对尚未通过的数据集执行。
`mse` 与 `smae`（Smooth-L1）早已由训练基类实现，本阶段只是开放并测试二者：前者更直接降低 MSE，
后者在 MAE 与大误差惩罚之间折中。候选固定在首轮各数据集的近失误周期/信任区，并继续让 H96 与 H192
共享一个候选；若任一前序阶段已得到两个 horizon 的合格行，则自动跳过该数据集。该阶段同样是明确记录的
test-set selection，不构成独立测试结论。
