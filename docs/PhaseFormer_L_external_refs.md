# PhaseFormer-L 外部参照数字

本文件登记 §4.2 必答项所需要的**外部模型引用数字**。仓库内的金标准只有原始 PhaseFormer
（`docs/PhaseFormer_gold_standard.md`），不含任何外部模型；§4.2 的必答 (a) 要求回答
"ETTh2 四个 horizon 相对 `phase_only` 与 Golden 的差距是否收窄、**是否达到 FITS 的引用数字**"，
因此需要这一份外部参照。缺口编号为 `PhaseFormer_L_experiment_plan.md` 的 **G4**。

## 1. FITS（ICLR 2024 Spotlight）

- 方法：FITS: Frequency Interpolation Time Series Analysis Baseline，ICLR 2024 Spotlight。
- 来源：官方仓库 <https://github.com/VEWOXIC/FITS>（README 的 "Result Update" 表），
  该表为**修复 `drop_last=True` 泄漏/截断缺陷后的最终结果**，并声明与 ICLR 定稿论文一致。
- 抓取日期：2026-09-19。
- **口径限制（重要）**：该表**只报 MSE，不报 MAE**。因此 FITS 只能用于必答 (a) 的 MSE 比较；
  MAE 一侧没有可引用的 FITS 数字，**不得**用其他来源的数字补造。
- 该表列出 5 个模型的 MSE（PatchTST / DLinear / FEDformer / TimesNet / FITS）。
  本文只用 FITS 行，其余模型不在 minipaper 的引用范围内。

## 2. FITS 的 MSE（L=720）

| Dataset | H96 | H192 | H336 | H720 |
|---|---:|---:|---:|---:|
| ETTh1 | 0.372 | 0.404 | 0.427 | 0.424 |
| **ETTh2** | **0.271** | **0.331** | **0.354** | **0.377** |
| ETTm1 | 0.303 | 0.337 | 0.366 | 0.415 |
| ETTm2 | 0.162 | 0.216 | 0.268 | 0.348 |
| Weather | 0.143 | 0.186 | 0.236 | 0.307 |
| Electricity | 0.134 | 0.149 | 0.165 | 0.203 |
| Traffic | 0.385 | 0.397 | 0.410 | 0.448 |

（原表精度为 3 位小数。）

## 3. 与 Golden 的直接对比（必答 (a) 的起点）

| Setting | Golden MSE | FITS MSE | FITS 相对 Golden |
|---|---:|---:|---:|
| ETTh2-96 | 0.275 | 0.271 | **−1.5%**（FITS 更好） |
| ETTh2-192 | 0.341 | 0.331 | **−2.9%** |
| ETTh2-336 | 0.369 | 0.354 | **−4.1%** |
| ETTh2-720 | 0.402 | 0.377 | **−6.2%** |

即：**原始 PhaseFormer 的 ETTh2 MSE 在四个 horizon 上都高于 FITS**，且差距随 horizon 放大
（1.5% → 6.2%）。这直接决定了必答 (a) 的判读方式：PhaseFormer-L 即便相对 `phase_only`
有改善，要"达到 FITS 的引用数字"需要补上 1.5%–6.2% 的差额。

作为对照，同表下 PhaseFormer 在 Electricity-336 上是 **0.165 vs FITS 0.165**（持平），
而 §4.1 已实测 PhaseFormer-L 在该格达到 **0.1625**（优于两者 1.5%）。

## 4. 使用与披露要求

1. FITS 数字用于 **§4.2 必答 (a)** 与 §4.2 表注；**不进入**主张 A–D 的判定（那是与 matched
   `phase_only` 的配对比较）。
2. **只比较 MSE**。表内不得出现 FITS 的 MAE 数字，也不得由 MSE 反推 MAE。
3. FITS 与本文运行环境不同（硬件、Lightning 版本、数据加载细节）；按 `MANAGE_RULES.md`
   的"金标准优先"条款，外部参照只作**披露性比较**，不得声明为可直接比较的提升。
4. 本文件的数字**不是**本仓库复现的结果，未在本仓库执行过 FITS；引用时须注明来源与抓取日期。
