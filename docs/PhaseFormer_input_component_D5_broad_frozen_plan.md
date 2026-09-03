# PhaseFormer 输入成分 D5：广泛冻结利用验证

> 状态：计划已冻结，待执行。固定 ETTm1、lookback=720、horizon=192、seed=2021、validation-only；
> 不训练、不读取 test。

## 目的

D4 显示“remove-trained 更能恢复”与“增强分支实际使用 A”不可混为一谈。D5 因此用同一 full-trained
checkpoint，对已完成的当前定义 D1/D2/D3 成分做一次广泛而低成本的即时利用筛查，而不新增候选或重训。

## 覆盖范围

|家族|条件|remove 定义|
|---|---:|---|
|D1|6|窗口内 Gaussian notch：96、48、32、24、677.647、205.714步，`sigma=1/720`|
|D2|4|最后24、48、96、192个标准化输入步直接置零|
|D3|5|global-linear、recent-linear、cycle-levels、phase-drift、cycle-amplitude；均按原定义末值锚定|

发现阶段从 validation 时间均匀抽取512个 origins；每一条件都测 full-trained 的 M0/M1/M2 从 `X` 到
remove 输入的 MAE/MSE 变化及 paired moving-block 95% CI。对 M1/M2 还测固定 full-input phase 预测和
融合系数、只替换 NLinear-style branch 输出的反事实。
反事实必须精确回放实际融合输出（最大绝对误差 < `2e-5`）才有效。

## 判读与停止规则

本轮只筛查，不把任一结果直接命名为目标 A。

1. M0 remove 效应明显为正：原版已使用该成分，不满足“原版未充分利用”的严格必要条件；
2. M1/M2 的 NLinear-only 反事实显著变差：分支实际使用了被扰动信息；
3. M1/M2 remove 损失较 M0 小：仅说明整体即时鲁棒性较强，不能推导分支不依赖该成分；
4. 只有 M0 接近零、M1/M2 的全模型与 NLinear-only 效应均稳定为正，才以 `--max-samples 0` 做完整
   validation 复核；复核通过才值得进入新的候选构造和多 seed 确认。若现有15项都未满足，则停止对
   这一候选库的扩展，不再做高耗时重训。

## 复现

```bash
/home/wangjing/miniconda3/envs/raft/bin/python scripts/run_d5_broad_frozen_utilisation.py \
  --output-dir research_runs/d5_broad_frozen_utilisation_control \
  --original-checkpoint <M0_FULL_CKPT> \
  --weak-checkpoint <M1_FULL_CKPT> \
  --rcrf-checkpoint <M2_FULL_CKPT> --require-cuda
```
