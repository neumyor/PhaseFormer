# M3 相对原始 PhaseFormer：样本分析协议

本协议在回放配对预测前固定，用于补充 `PhaseFormer_M3_multi_anchor_paper_draft.md`。

- 对象：M3 与同代码、同切分重跑的原始 PhaseFormer。
- setting：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity；L720/H96、30% train、8 epoch、seed 2021、Huber、validation only。
- 成功样本：sample×channel 的相对 MSE 不高于 -10%，且 MAE 同时下降。
- 失败样本：相对 MSE 不低于 +10%，且 MAE 同时上升；其余均放入“其他”，避免单指标挑选。
- 案例：在每个数据集、每个组内按绝对 MSE 差排序，同一通道相距不足 96 的滑窗去重；每组保留 5 例，论文图每组展示 3 个不同数据集代表。
- 输入侧特征：近期漂移、lag-24 相关、差分波动、相位可靠度。
- 事后特征：未来水平迁移、未来 lag-24 相关、未来差分波动。它们只用于解释，不能当作模型可见信息。
- 中间量：A1/I0/R0 平均权重、路由熵、三个锚点预测分歧。
- 统计：逐数据集比例、输入形态分层、成功/失败组标准化均值差；总体 MSE 另用块长 96 的 paired block bootstrap 描述区间。滑窗不视为独立样本，不报告普通 i.i.d. t 检验。
- 选择偏差：M3 已由当前 validation 选出，本分析仅定位优势与边界，不作为独立泛化证明；matched rerun 也不替代 Golden。

生成脚本：`scripts/analyze_m3_vs_original.py`。完整本地证据目录：`research_runs/m3_vs_original_phaseformer_v1/`。
