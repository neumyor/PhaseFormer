# Strict-T28 在 ETTh1 的正式 Golden 对比

## 目的与边界

应用户要求，直接评估单阶段 `pctf_anchor_repair_strict_t28` 在 ETTh1 的 H96/H192。该模型是“完整
A2（PhaseFormer + LFF-NLinear + RCRF）预测为锚点，再添加有界 ICPT 周期间水平/周期内形状修正”的
单 checkpoint 模型。

这不是重新筛选参数：ETTh1 预先采用共享 `cycle_period=48`，并固定 T28 的
`correction/deformation/global-level=0.60/0.24/0.12`。由于本轮按用户指令读取了 test，后续若依据
这些数值改变 ETTh1 参数，必须标记为 test-set selection；本表本身只报告冻结配置的首次 test。

## 协议

- 输入 720；输出 H96、H192；100% train；Huber；最多 30 epoch；best-validation checkpoint。
- seeds：2021、2022、2023；每个 checkpoint 只读取一次 test。
- 一次性联合训练，不使用 A2 预训练 checkpoint；composer 的 A2-derived 输入均 stop-gradient，A2
  只通过 anchor loss 训练。
- GPU：RTX 4090；`required_cuda=true`；测试输出在
  `research_runs/pctf_strict_t28_etth1_formal_v1/`（被 gitignore，未提交）。
- Golden 取自 `docs/PhaseFormer_gold_standard.md`。Delta 定义为 `T28 mean - Golden`；正值为退化。

## 结果

| Setting | 三 seed test（MSE / MAE） | T28 mean±sample std | Golden | Delta（绝对；相对） | 判断 |
|---|---|---:|---:|---:|---|
| ETTh1-H96 | 2021: 0.370117 / 0.397449；2022: 0.365599 / 0.394854；2023: 0.364953 / 0.393916 | 0.366890±0.002813 / 0.395406±0.001830 | 0.359 / 0.382 | +0.007890；+2.198% / +0.013406；+3.510% | 双指标退化 |
| ETTh1-H192 | 2021: 0.397871 / 0.414462；2022: 0.402255 / 0.416912；2023: 0.401141 / 0.415637 | 0.400422±0.002279 / 0.415671±0.001225 | 0.397 / 0.404 | +0.003422；+0.862% / +0.011671；+2.889% | 双指标退化 |

结论：冻结的 strict-T28 在 ETTh1 的两个测试长度均未超过 Golden，且三个 seed 都没有出现双指标胜出。
H192 的 MSE 只高 0.862%，但 MAE 高 2.889%；H96 两项退化更明显。因此，这一结果不支持把当前 T28
直接推广为 ETTh1 的默认方案，也不能与此前两阶段 Full Repair 在 ETTh2/ETTm2 的正向结果混为一谈。
