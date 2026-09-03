# D7 内部路径诊断

固定 ETTm1/L720/H192/seed2021、512个均匀 validation origins；无训练、无 test。对 M1/M2 的完整输入前向，记录每个窗口的 phase MAE、fused MAE、NLinear MAE 和 correction 与 phase residual 的对齐度。目标为 `phase_MAE-fused_MAE`，即 NLinear 融合实际带来的收益。

预先固定六个历史描述量：近期差分、近期偏离、周期水平波动、周期幅度波动、最后周期水平偏移、日滞后变化。用连续五折 OOF ridge 的 R²及特征与收益的相关性判断它们能否预测收益；仅作候选发现，不可替代输入干预证明。
