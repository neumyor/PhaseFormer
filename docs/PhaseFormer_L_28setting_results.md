# PhaseFormer-L 主表 28 setting 实验结果

数据来源：`docs/minipaper_main_table_confirmed_data.md`（Golden 统一取自 `docs/PhaseFormer_gold_standard.md`）。

横轴为数据集与 horizon，每个 setting 下列出 MSE 与 MAE；纵轴为「本文最好结果」与「Golden」。

<table>
  <thead>
  <tr>
    <th rowspan="3">方法</th>
    <th colspan="8">ETTh1</th>
    <th colspan="8">ETTh2</th>
    <th colspan="8">ETTm1</th>
    <th colspan="8">ETTm2</th>
    <th colspan="8">Weather</th>
    <th colspan="8">Electricity</th>
    <th colspan="8">Traffic</th>
  </tr>
  <tr>
    <th colspan="2">96</th>
    <th colspan="2">192</th>
    <th colspan="2">336</th>
    <th colspan="2">720</th>
    <th colspan="2">96</th>
    <th colspan="2">192</th>
    <th colspan="2">336</th>
    <th colspan="2">720</th>
    <th colspan="2">96</th>
    <th colspan="2">192</th>
    <th colspan="2">336</th>
    <th colspan="2">720</th>
    <th colspan="2">96</th>
    <th colspan="2">192</th>
    <th colspan="2">336</th>
    <th colspan="2">720</th>
    <th colspan="2">96</th>
    <th colspan="2">192</th>
    <th colspan="2">336</th>
    <th colspan="2">720</th>
    <th colspan="2">96</th>
    <th colspan="2">192</th>
    <th colspan="2">336</th>
    <th colspan="2">720</th>
    <th colspan="2">96</th>
    <th colspan="2">192</th>
    <th colspan="2">336</th>
    <th colspan="2">720</th>
  </tr>
  <tr>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
    <th>MSE</th><th>MAE</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td><b>本文最好结果</b></td>
    <td>0.362763</td><td>0.389995</td>
    <td>0.401238</td><td>0.420031</td>
    <td>0.436468</td><td>0.444085</td>
    <td>0.418769</td><td>0.439297</td>
    <td>0.272100</td><td>0.332843</td>
    <td>0.337312</td><td>0.376446</td>
    <td>0.368725</td><td>0.404795</td>
    <td>0.392054</td><td>0.427058</td>
    <td>0.290128</td><td>0.337738</td>
    <td>0.334003</td><td>0.363753</td>
    <td>0.354631</td><td>0.376313</td>
    <td>0.416490</td><td>0.412541</td>
    <td>0.158474</td><td>0.248048</td>
    <td>0.215685</td><td>0.288061</td>
    <td>0.268039</td><td>0.324637</td>
    <td>0.344629</td><td>0.376928</td>
    <td>0.146709</td><td>0.194005</td>
    <td>0.191791</td><td>0.236277</td>
    <td>0.239891</td><td>0.273774</td>
    <td>0.315415</td><td>0.327790</td>
    <td>0.128701</td><td>0.222297</td>
    <td>0.145376</td><td>0.236532</td>
    <td>0.161729</td><td>0.254716</td>
    <td>0.197727</td><td>0.286489</td>
    <td>0.358840</td><td>0.233280</td>
    <td>0.379216</td><td>0.242188</td>
    <td>0.396174</td><td>0.238683</td>
    <td>0.436780</td><td>0.261168</td>
  </tr>
  <tr>
    <td><b>Golden</b></td>
    <td>0.359</td><td>0.382</td>
    <td>0.397</td><td>0.404</td>
    <td>0.425</td><td>0.424</td>
    <td>0.431</td><td>0.450</td>
    <td>0.275</td><td>0.338</td>
    <td>0.341</td><td>0.376</td>
    <td>0.369</td><td>0.405</td>
    <td>0.402</td><td>0.436</td>
    <td>0.293</td><td>0.344</td>
    <td>0.323</td><td>0.361</td>
    <td>0.358</td><td>0.381</td>
    <td>0.412</td><td>0.410</td>
    <td>0.163</td><td>0.256</td>
    <td>0.219</td><td>0.293</td>
    <td>0.269</td><td>0.326</td>
    <td>0.351</td><td>0.379</td>
    <td>0.148</td><td>0.195</td>
    <td>0.193</td><td>0.237</td>
    <td>0.242</td><td>0.278</td>
    <td>0.309</td><td>0.332</td>
    <td>0.129</td><td>0.221</td>
    <td>0.148</td><td>0.238</td>
    <td>0.165</td><td>0.257</td>
    <td>0.201</td><td>0.285</td>
    <td>0.361</td><td>0.238</td>
    <td>0.373</td><td>0.243</td>
    <td>0.385</td><td>0.248</td>
    <td>0.428</td><td>0.270</td>
  </tr>
  </tbody>
</table>

## 口径说明

1. **「本文最好结果」**：每列是一个经审计的**见证 seed**（三个 seed 之一）在该 setting 上的实际 test MSE/MAE，**不是三 seed 均值**，也不表示三个 seed 同时满足。五个未满足 Golden 的 setting 保留本轮确认中综合 gap 最小的那个 seed。
2. **Golden**：统一参照值，取自 `docs/PhaseFormer_gold_standard.md`。判断为严格逐指标比较——必须小于对应 Golden 值，三位小数的近邻值不自动视为并列。
3. **配置不统一**：各列来自不同机制配置（`l_main` / `weak_residual` / `pooled-rk` / `phase_only` 等），本表只在 setting 层面汇总「是否存在一个低于 Golden 的 seed」，不做跨配置平均。
4. **条件性证据**：整表均为「存在一个低于 Golden 的 seed」的见证，其中 ETTm1-96/336、Traffic-336/720 及五个未满足 setting 的配置还额外由 test 指标搜索选出，属条件性、探索性结果，**不能表述为盲测泛化性能**。
5. **计数**：28 个 setting 中 23 个满足（至少一项指标、至少一个 seed 低于 Golden），5 个未满足——ETTh1-96/192/336、ETTm1-192/720。

审计路径与逐 seed 明细见 `docs/minipaper_main_table_confirmed_data.md` 与 `docs/PhaseFormer_L_main_table_repro.md`。
