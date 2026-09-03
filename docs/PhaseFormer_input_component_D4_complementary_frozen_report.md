# PhaseFormer 输入成分 D4：互补信息冻结诊断

> 状态：已完成。范围固定为 ETTm1、lookback=720、horizon=192、seed=2021、validation-only。
> 这是一次冻结 checkpoint 的前向诊断，**没有训练，也没有读取 test**。它的目的不是建立新效果
> 声明，而是补足 D1--D3 无法回答的“增强分支实际在用什么”问题。

## 1. 问题与设计

此前的 D3 是 remove-trained：`X-A` 上重新训练后，M1/M2 常比 M0 恢复得更好。它只能说明增强模型
可用剩余信息补偿缺失，并不能说明 NLinear 分支在完整模型中使用 A，或只使用 A 以外的信息。

本轮只选择 D3 中差异最大的两个、且可逐窗口精确提取的轨迹：

|成分 A|定义|选择理由|
|---|---|---|
|`recent_linear`|最后96步估计的线性方向，向全720步延伸，末点为零|D3 remove-trained interaction 最强：M1 -5.55pp、M2 -4.39pp|
|`cycle_levels`|每个24步块均值相对最后块均值的轨迹|D3 remove-trained interaction 强且 M1/M2 一致：-3.16/-3.21pp|

对每个 full-trained checkpoint（M0 original、M1 weak residual、M2 RCRF+NLinear plain）在完整 validation
评估三种输入：

|视图|输入|回答的问题|
|---|---|---|
|full|`X`|匹配锚点|
|remainder|`X-A`|已有模型是否立即依赖 A；其余历史 B 是否足够|
|component-anchor|`repeat(last(X))+A`|A 加上共同末值锚点是否足够；它不是 `X=A+B` 的代数互补|

最后一个定义特意保留最后输入值。D3 的 A 在末点为零，而 NLinear-style head 把最后值作为 persistence
anchor；若直接输入 A，会把“没有 A”与“没有锚点”混为一谈。

对于 M1/M2，另做分支反事实：固定 full 输入时的相位预测与融合系数，只以对应视图重新计算
NLinear-style branch，再重组输出。若这项反事实变差，才可称“该 NLinear 分支实际使用了被改变的
输入”。每个 MAE/MSE 相对变化均以完整 validation 的 paired moving-block bootstrap（1000次，块长192）
给出95% CI。

## 2. 完整模型对输入视图的即时反应

下表为相对各模型 full MAE 的变化，正数为变差；括号为95% CI。原始 MAE 分别为 M0=0.458644、
M1=0.461456、M2=0.462057。

|A|视图|M0 original|M1 weak residual|M2 RCRF+NLinear|
|---|---|---:|---:|---:|
|recent-linear|`X-A`|+140.1% [125.7,156.0]|+201.0% [179.2,224.6]|+172.4% [153.9,193.4]|
|recent-linear|`last(X)+A`|+154.1% [137.4,172.2]|+223.7% [201.2,248.4]|+193.1% [173.1,214.8]|
|cycle-levels|`X-A`|+51.8% [43.6,61.0]|+46.0% [37.9,55.0]|+44.8% [37.1,53.5]|
|cycle-levels|`last(X)+A`|+41.9% [35.5,49.4]|+40.3% [34.1,47.5]|+37.4% [31.5,44.2]|

两种隔离视图都远差于 full，故 A 单独不充分、B 单独也不充分；它们更适合作为“模型依赖的方向”而非
可直接部署的缺失数据设定。

## 3. 仅替换 NLinear 分支的反事实

下表为 `phase_full + alpha_full * nlinear_changed` 相对 full 的 MAE 变化。所有分支重组的最大回放误差
小于 `4.8e-6`，故反事实公式与实际融合前向一致。

|A|视图|M1 weak residual|M2 RCRF+NLinear|解释|
|---|---|---:|---:|---|
|recent-linear|`X-A`|+222.3% [204.9,242.4]|+31.2% [28.4,34.4]|M1 的 NLinear branch 明确使用 A；M2 分支也受影响，但量级远小于 M2 完整输出反应|
|recent-linear|`last(X)+A`|+225.4% [202.8,249.9]|+77.2% [66.0,89.8]|同上；A 与 anchor 不足以恢复完整分支所需上下文|
|cycle-levels|`X-A`|+33.3% [26.5,41.4]|+30.1% [24.2,36.8]|两个 NLinear branch 都实际使用 A|
|cycle-levels|`last(X)+A`|+11.4% [9.3,13.9]|+21.9% [18.0,26.4]|同样是正且 CI 不跨零，但影响小于 `X-A`|

## 4. 结论：哪些推断被支持，哪些未被支持

1. **NLinear branch 的确会实际使用这两类轨迹。** 四组 M1/M2 分支反事实均显著变差，不能再把
   D3 的现象简单解释成“增强分支完全不依赖 A”。
2. **`recent-linear` 不是目标盲区 A。** M0 在 `X-A` 时也立即大幅变差（+140.1%）；M1/M2 更大的
   完整模型反应，说明它们的组合输出更依赖这个轨迹，却不能证明原版没有用它。
3. **`cycle-levels` 说明“分支使用”和“整体鲁棒性”必须分开。** NLinear branch 对它有显著反事实
   损失（M1 +33.3%、M2 +30.1%），但完整 M1/M2 在 `X-A` 时的损失略小于 M0（46.0/44.8% vs 51.8%）。
   这与 D3 remove-trained 的“增强更可恢复”方向一致：增强分支使用 A，同时还能用其他信息抵消一部分
   A 缺失。
4. 因此，目前不应使用“增强版本在扰动下更鲁棒，所以它使用的是扰动相反的信息、而原版没有充分使用”
   作为论文因果链。正确说法是：**增强路径可同时使用被删轨迹与其余上下文；其缺失恢复优势不是
   ‘不依赖 A’ 的证据。**

## 5. 可复现性与范围

- 代码提交：`fe2711d`；执行环境：`raft` conda + CUDA；运行时约22秒。
- 入口：

```bash
/home/wangjing/miniconda3/envs/raft/bin/python scripts/run_d4_complementary_frozen_probe.py \
  --output-dir research_runs/d4_complementary_frozen_probe_control \
  --original-checkpoint <M0_FULL_CKPT> \
  --weak-checkpoint <M1_FULL_CKPT> \
  --rcrf-checkpoint <M2_FULL_CKPT> --require-cuda
```

- 可审计的结果：`research_runs/d4_complementary_frozen_probe_control/` 下的
  `frozen_complementary_results.csv`、`paired_sample_effects.npz`、`protocol.json`。这些为 gitignore 的
  运行产物；本报告中的数字均可由 CSV 复核。
- 单数据集、单 horizon、单 seed、validation-only，不报告泛化或正式显著性结论。
