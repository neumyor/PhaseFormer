# E18 / minipaper §4.6 — 负对照补做（plan）

Artifacts（两个新脚本，均不改动任何既有文件）：

* `scripts/phaseformer_L/e18_negative.py` — 行 1 / 行 5 的训练 runner（78 runs）；
* `scripts/phaseformer_L/e18_svd_truncation.py` — 行 3 的 28-setting 截断分析（0 训练）。

Output root: `research_runs/phaseformer_L_e18_negative_v1/`（`--output-root` 显式传入，
不与其它 E 单元共用根目录；schedule §3.1 的硬性要求）。

---

## 1. §4.6 要求什么

`docs/PhaseFormer_L_minipaper.md:503-511` 是负对照汇总表（§4.6），5 行；其中 3 行标"本文补做"：

| 行 | 要求 | 新训 | 本单元 |
|---|---|---:|---|
| 行 1 | 输入平滑在 **PhaseFormer-L** 上复测 **2 档**，域 = **7 个 test-selected setting** × 3 seed | **42** | `--stage smooth` |
| 行 3 | SVD 截断 vs 秩约束训练扩到 **28 setting**（分析，读既有 checkpoint） | 0 | `e18_svd_truncation.py` |
| 行 5 | 边界消融 `pooled_lowrank` 绝对 `rank∈{1,2}`，**6 setting** × 3 seed × 2 rank | **36** | `--stage rank12` |
| 行 2 / 行 4 | 结构化坐标、q=1/32 容量 | 0 | **不补做**，minipaper 保持 "—" |

`docs/PhaseFormer_L_execution_schedule.md:163-175`（§2.3 E18 明细）与决策 **D-4**
（`schedule:21`：行 1 复测范围 = 7 个 test-selected setting、2 档）是本单元的排期依据。
合计 **78 runs**，与 `schedule:65`（E18 行）一致。

---

## 2. 继承的定义（不重新发明）

| 量 | 来源 | 位置 |
|---|---|---|
| 协议常量 `LOOKBACK/PERIOD/MAX_EPOCHS/LOSS/PERCENT/SEEDS` | **import** `e14_main_matrix` | `scripts/phaseformer_L/e14_main_matrix.py:57-66` |
| 7 个 test-selected setting | **import** `REUSE_SETTINGS_FULL` | `e14_main_matrix.py:92-97` |
| "哪个 run 是 `l_main`" | **import** `_arm_match` | `e14_main_matrix.py:213-239` |
| 新格超参 `gate_init=0.2`/`lr=1e-3` | **import** `NEW_CELL_GATE_INIT`/`NEW_CELL_LR`（D-2） | `e14_main_matrix.py:65-66` |
| 6-setting E8 block（行 5 域） | `REUSE_SETTINGS_PHASE_ONLY` 的同一 6 setting（ETTh2-96/720、ETTm2-96/192、Weather-96/192） | `e14_main_matrix.py:99-101`；`schedule:79-80` |
| 平滑算子的 boxcar 旋钮 | `weak_period_residual_smooth_ratio` | `scripts/run_smooth_ratio_sweep.py:37`；`src/models/PhaseFormer.py:1054-1056`；`src/models/phase_adapters.py:113-115` |
| 平滑算子的 causal-EMA `alpha` 旋钮 | `weak_period_residual_causal_ema_alpha` | `scripts/run_causal_ema_smooth_sweep.py:39`；`src/models/PhaseFormer.py:1057-1059` |
| 绝对秩旋钮 | `weak_period_residual_rank`（+`pool_factor=1`） | `src/models/PhaseFormer.py:1013-1016`；E14 的写入方式 `e14_main_matrix.py:437-439` |
| SVD 截断代数 | `U[:, :r] @ diag(S[:r]) @ Vh[:r, :]` | `scripts/analyze_weak_residual_svd_truncation_eval.py:41-44` |
| 评测指标定义 | `scripts/search_phaseformer.py::evaluate` 的 abs/sq 累加口径 | `scripts/search_phaseformer.py:478-506` |
| 每 GPU 一 run + 重试 + JSON manifest 的 dispatch | E14 | `e14_main_matrix.py:487-538` |

**没有改动任何既有数值口径**；`e18_svd_truncation.py` 直接 import E14 的 `ARMS`/`_arm_match`，
"哪个 run 是 `l_q1_8`"因此在全仓库只有一处定义。

---

## 3. 行 1：两档平滑的精确构造

两档取值由 schedule §2.3 在 E18 冒烟前**冻结**（`schedule:173-175`），本单元不重新选档：

| `--levels` | head | `weak_period_residual_smooth_ratio` | `weak_period_residual_causal_ema_alpha` |
|---|---|---|---|
| `boxcar` | `shared`（稠密 PhaseFormer-L） | `0.5` | **不写入**（preset 默认 0.08 生效） |
| `causal_ema` | `shared` | `0.5` | **`0.08` 字面写入** |

两档因此是**同一个算子、同一 `smooth_ratio`，唯一差别是 alpha 是否被显式 pin**，
结果可解释为"causal EMA 混合 50%"与"同式默认 alpha 50%"的对照。

### 3.1 一处必须披露的事实：两个"档"的操作符其实是同一个

`WeakPeriodResidualHead.forward` 的平滑分支是 **causal EMA**，不是 boxcar：

```
centered = x - last
if smooth_ratio: smoothed = _causal_ema(centered, alpha); centered = (1-sr)*centered + sr*smoothed
```
（`src/models/phase_adapters.py:110-123`；`_causal_ema` 在
`src/models/asymmetric_trend_components.py:116-125`，默认 `alpha=0.08`，且 0.08 也是
`PhaseFormer.py:1057-1059` 的读键默认值。）

而在 `pooled_lowrank` 头上，`smooth_ratio` 配的算子是 **boxcar**
（`F.avg_pool1d`，`src/models/phase_adapters.py:168-179`），窗口 `smooth_window=24`。
既有 boxcar 扫描（`scripts/run_smooth_ratio_sweep.py`）跑的是后者，既有 causal EMA 扫描
（`scripts/run_causal_ema_smooth_sweep.py`）跑的是**全秩 shared 头**——与本行同一算子。

**因此 §4.6 行 1 的"2 档"在 PhaseFormer-L（`shared` 头）上并不是两种不同算子**，
而是同一 causal-EMA 算子的两档强度：`boxcar` 档 = `smooth_ratio=0.5` + 默认 alpha 0.08，
`causal_ema` 档 = 同一 `smooth_ratio` + 显式 alpha 0.08。脚本据此**不把两档称作
"boxcar 算子"**：`level_key` 记为 `boxcar_s0.5` / `causal_ema_s0.5_a0.08`，
`head_type=shared` 与 `causal_ema_alpha` 两列逐行写入 `results.csv`，表注可据此如实披露。
若后续希望得到真正的 boxcar 算子档，唯一的做法是改用 `pooled_lowrank` 头
（`run_smooth_ratio_sweep.py` 的既有配置），但那不是 PhaseFormer-L 的主工作点，本单元不做。

### 3.2 复用 vs 新训（必须写进表注）

* 行 1 的 **42 个格全部是新训**：给定 setting 上，"`shared` 头 + `smooth_ratio=0.5`"这一配置
  在 §4.2 的复用清单里不存在（`_arm_match` 明确要求 `smooth_ratio == 0`，
  `e14_main_matrix.py:224-225`；`audit_top2_direction_retention_reuse.py:99` 的审计同样要求 0）。
* 42 格**全部按 D-2 新格超参**训练：`weak_period_residual_gate_init=0.2`、`learning_rate=1e-3`
  （`e14_main_matrix.py:65-66`、`schedule:19`、`schedule:29` 的新格行）。
  脚本对每一格核验这两个值（`config_matches`），不符即 `--verify` 失败。
* 但"同 setting 的 §4.2 基线"里，21 个 `l_main` 格是**复用格**，携带 **Stage-0 冻结**
  `(gate, lr)`（例如 ETTm2-96 gate=0.5、lr=3e-4；Weather-96/192、ETTh2-96/720、
  Electricity-336 gate=0.5、lr=1e-3——逐 setting 取值见 `schedule:31` 与
  `e18_negative_manifest.json` 的 `baseline` 块）。因此"平滑格 vs `l_main` 基线"的配对
  **跨了两套超参协议**，只能做 setting 内配对诊断，不是同协议对照；表注须逐格披露。
* `--baseline-manifest` 指向 E14 的 `stage_a_manifest.json` 时，脚本把每个平滑格对应的
  `l_main` 基线（`status`、`run_dir`、`gate_init`、`learning_rate`）写进
  `results.csv` 的 `baseline_*` 列，使上面这条披露可审计。该选项**只记录来源**，
  不改变训练内容；E14 manifest 条目若带 `--evaluate-test` 或 `_arm_match` 不通过则被拒绝并记录原因。
* 与 §4.6 既有列的关系：既有的 "14/14 组合无一改善" 来自 `pooled_lowrank` 头 × 5 档
  （boxcar 轮）+ shared 头 × 5 档（causal EMA 轮）。行 1 复测的是 PhaseFormer-L
  （`shared`）在 2 档上的结论，**不是**既有 14 个组合的子集，也不是它的重算。

---

## 4. 行 5：绝对秩 1/2 的边界消融

* 机制 = `weak_residual`，头 = `pooled_lowrank`，`weak_period_residual_pool_factor=1`，
  `weak_period_residual_rank ∈ {1, 2}`（**绝对**秩，不是 `H/div`）。
* 域 = E8 的 6 个 setting（ETTh2-96/720、ETTm2-96/192、Weather-96/192）；Electricity-336
  被排除，因为 §4.6 行 5 的注册消融就定义在这 6 个上（`schedule:169`）。
* 36 格全部新训、全部按 D-2 新格超参（`gate_init=0.2`、`lr=1e-3`）；`smooth_ratio=0.0`
  显式写入，避免与行 1 的平滑格混淆（脚本核验该值）。
* **必须在表注声明的三点**：
  1. 绝对秩 1/2 **落在既有低秩网格之外**——已测最深的网格点是 `q=1/32`，
     对应秩 3/6/10/22（`minipaper:300`；`docs/PhaseFormer_rank_sweep_conditioned_experiment.md` §7.3）。
      §4.6 把这一行登记为"网格之外"的边界消融，预期退化，幅度上界由
     `1 − capture(1/2)`（秩 1 放弃 14%–35%、秩 2 放弃 3%–18% 的支路价值）经 `g²` 折算
     （`minipaper:511`、`minipaper:301`）。
  2. **不得把该结果读作"秩 2 必要"的正反证据**：`docs/PhaseFormer_L_experiment_plan.md` §9.5
     （`experiment_plan:304`）明写，若增量落在 seed 噪声内，只能支持命题 2 的秩-1 预测，
     不能反向宣称秩-2 无效。脚本与表注均保留该约束。
  3. 该行的参照是 E14 的 `l_main`（稠密 `shared`，§4.2 主行）与 `l_q1_4`/`l_q1_8`，
     其中后两者是**秩约束训练**的已测点，用来给 r=1/2 的退化幅度定位；脚本不新训任何对照。

---

## 5. 行 3：28-setting 截断分析

### 5.1 读什么

| 角色 | 来源 | 位置 |
|---|---|---|
| 全秩训练头 | E14 `l_main` 格（`weak_residual` + `shared`） | `stage_a_manifest.json` 的 `cells[arm=l_main]`；复用格指向 E3 lineage run dir |
| 秩约束训练头 | E14 `l_q1_4`、`l_q1_8` 格（`rank=H/4`、`H/8`，`pool_factor=1`） | 同上 |
| 全秩权重 | `weak_period_residual.linear.weight`（`H×720`） | run 的 best-val checkpoint（`metrics.csv` 的 `checkpoint` 列） |

解析规则：manifest 条目仅在**自身命令不含 `--evaluate-test`** 且 `_arm_match` 通过时才被采用；
run dir 优先取 `source.run_dir`（复用格），否则在 `--output-dir`/`--e14-root` 的 `runs/*/config.json`
里按 `(dataset, horizon, seed)` + arm 指纹匹配（新格——E14 manifest 不记录新格的 run dir，
因为 run id 里带 config hash）。每个 run 的 checkpoint 以它自己 `metrics.csv` 记录的
repo-relative 路径为准，`attempts/*/checkpoints/` 只在记录路径失效时作为兜底。

### 5.2 截断秩的选择（决定 + 理由）

* **主秩固定 `r=10`（`--ranks` 默认 `"10"`）**，逐 setting 都算。理由：`r=10` 是 E11 实测
  Electricity-336 反例的那个秩（截断 +29% vs 训练 +0.7%，`minipaper:509`、
  `PhaseFormer_lowrank_mechanism_analysis.md` §3.2），固定它使 minipaper 引用的那一个量
  在 28 个 setting 上同秩可比；E14 的 `l_q1_4/q1_8` 网格在各 setting 上秩不同
  （H=96 → 24/12；H=720 → 180/90），无法做跨 setting 的同秩对照。
* **每个 setting 同时报告它自己的 `rank=H/4`、`H/8`（`native_rank_q1_4`/`native_rank_q1_8` 列）**，
  并给出 `trained_lowrank_arm`/`trained_lowrank_rank`：即与该 setting 的 `r=10` 截断结果
  **同 seed、同 setting** 配对的那个已训低秩格（两格中秩更接近 10 者，通常为 `l_q1_8`）。
* **诚实披露的结构性错配**：`r=10` 在 H=720 上等于已训最低秩的 1/9（90），在 H=96 上却**高于**
  已训秩 12/24。因此 `trained_lowrank_rank` 列逐行给出实际配对秩，`gap_*` 必须在同秩语境下读；
  需要更多秩时用 `--ranks 10,12,24,...` 追加评测。
* 报告列（`svd_truncation_table_28.csv`，28 行）：`truncated_mse/mae`、`trained_lowrank_mse/mae`、
  `gap_truncated_vs_trained_mse/mae_pct`（= 100·(截断−训练)/训练）、`full_rank_mse/mae`、
  `gap_trained_vs_full_mse_pct`，外加 `primary_rank`、`trained_lowrank_arm/rank`、
  `native_rank_q1_4/8`、`seed`、`split`、`records_test`、以及来源列
  （`full_rank_run_dir`、`full_rank_cell_status`、`trained_lowrank_run_dir`、
  `trained_lowrank_cell_status`、`full_rank_gate_init`、`full_rank_learning_rate`）。
  逐 (setting, seed, rank) 的细表在 `svd_truncation_per_rank.csv`。

### 5.3 与 E11 的可比性（**必须写进表注**）

E11 的 Experiment 2 是 **test-based**：`scripts/analyze_weak_residual_svd_truncation_eval.py`
的 docstring 明写 "evaluate real test-set MSE/MAE"，并在 `build_model_and_loader` 里
`data_provider(exp_args.dataset_args, "test")`（`_weak_residual_analysis_common.py:149`，`data_provider(..., "test")`），
单 seed 2021、7 个 setting、checkpoint 来自 `research_runs/rank_sweep_2_stage1`。

本脚本的默认与唯一评测 split 是 **validation**（`--evaluation-split` 只接受 `val`/`validation`，
`test` 会被 argparse 拒绝；代码里没有任何 test loader）。因此：

* 本表数字与 E11 §3.2 的 7 行**不是同一口径，不能互相替代**；
  `e18_svd_truncation_summary.json` 的 `e11_comparability` 块逐条记录该差异，
  `svd_truncation_table_28.csv` 每行带 `split` 与 `records_test=false`；
* 两表**共用的定义**只有：截断代数、三路比较的结构（截断 / 训练低秩 / 全秩）；
  连 checkpoint 来源也不同（E14 `l_main` cells，§4.0 协议 + D-2/D-4，而非 E3 lineage）。
* 表注建议写法："28 setting 的截断-训练差距为 **validation** 口径（E11 的 7 setting 为 test 口径），
  两者不可直接相减；E11 的 Electricity-336 反例（r=10，+29% vs +0.7%）仍按 test 口径引用。"

---

## 6. test-split 保证

* `e18_negative.py`：构建的 argv 里**永不出现 `--evaluate-test`**（`command()` 里没有该分支），
  manifest 记录 `reads_test=false`、`evaluate_test_passed=false`，
  `results.csv` 记录 `test_mse_recorded`/`test_mae_recorded`（默认 False）作为审计列。
  test 的单次读取照 E14 的做法另立阶段（`scripts/phaseformer_L/e14_read_test.py` 是 E14 的实现，
  E18 的单次读阶段在 §4 schedule 的"阶段 4/5"里另行安排）。
* `e18_svd_truncation.py`：只构造 validation loader；`--evaluation-split {val,validation}`；
  输出每行 `records_test=false`；summary 记 `test_split_read=false`。

---

## 7. CLI 与产物

```text
# 行 1 + 行 5 训练（两 stage 可分开提交）
python scripts/phaseformer_L/e18_negative.py \
  --stage {smooth,rank12,all} --gpus 0,1,2,3,4,5,6,7 \
  --output-root research_runs/phaseformer_L_e18_negative_v1 \
  [--datasets A,B] [--seeds 2021,2022,2023] [--levels boxcar,causal_ema]
  [--ranks 1,2] [--num-workers 4] [--retries 1] [--poll-seconds 15]
  [--baseline-manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json]
  [--verify] [--dry-run]

# 行 3 分析
python scripts/phaseformer_L/e18_svd_truncation.py \
  --e14-root research_runs/phaseformer_L_e14_main_v1 \
  --output-root research_runs/phaseformer_L_e18_negative_v1 \
  [--manifest <path>] [--datasets ...] [--horizons ...] [--seeds 2021]
  [--ranks 10] [--evaluation-split val] [--max-eval-samples 0]
  [--num-workers 4] [--device {auto,cpu,cuda}] [--skip-figures]
  [--verify] [--dry-run]
```

`<output-root>/`（两侧共用根目录，文件名互不冲突）：

* `e18_negative_manifest.json` — 78 格清单（逐格 `overrides` + 完整 `command`）、
  协议块、`counts`、baseline 解析报告；`results.csv` — 每格一行（列见 §5.2 + `baseline_*`）；
* `e18_negative_summary_<stage>.json` — 训练完成/失败清单；
* `e18_negative_verify.json` — `--verify` 的逐格对账结果；
* `e18_svd_truncation_plan.json` — 28×seed 的解析结果（run dir / checkpoint / gate / lr）、
  `problems`、`e11_comparability`；
* `svd_truncation_per_rank.csv`、`svd_truncation_table_28.csv`；
* `e18_svd_truncation_summary.json`、`figures/svd_truncation_28.svg`（`--skip-figures` 可关）；
* `_logs/<cell>.log` — 每格训练日志。

**每个写入 `results.csv` / CSV 的列都由代码真实写出**：`results.csv` 的列由
`RESULTS_FIELDS` 常量与 `summarize()` 的键一一对应；截断分析的 `PER_RANK_FIELDS` 与
`evaluate_setting()` 的行字典、`TABLE_FIELDS` 与 `build_table()` 的行字典已逐项核对
（无声明未写、无写未声明；`build_table` 额外写了 `seeds_evaluated`，已在 `TABLE_FIELDS` 中声明）。

---

## 8. 判定口径与歧义（必须披露）

1. **"2 档"实为同一 causal-EMA 算子的两档**（§3.1）：`shared` 头上 `smooth_ratio` 配的算子
   是 EMA 而非 boxcar；真正的 boxcar 只在 `pooled_lowrank` 头上。表注须照此写，不可写成
   "boxcar 与 causal EMA 两种算子"。
2. **alpha 0.08 与 operator 默认值相同**：`_causal_ema` 的签名默认就是 0.08
   （`asymmetric_trend_components.py:116`），`PhaseFormer.py:1058` 的读键默认也是 0.08。
   因此 `causal_ema` 档与 `boxcar` 档在**数值上可能完全是同一个配置**——
   差别仅在 config.json 是否显式记录 `weak_period_residual_causal_ema_alpha`。
   这不会改变训练结果，但会让两档的 config hash **不同**（hash 覆盖全部 hyperparams），
   因此 42 个格仍然是 42 次合法训练；表注须说明两档"配置等价、hash 不同"，
   以免读者把它当作两个独立算子的重复证据。若用户希望两档必须数值不同，
   需要另选 alpha（例如 0.04 或 0.16），但那会违反"取值已冻结、不得改动"的约束，本单元不自行改档。
3. **行 1 的 42 格是"setting 级已见过的配置协议"**：7 个 setting 来自 test-set selection，
   且其 `l_main` 基线可能是复用格（Stage-0 冻结超参），配对比较跨协议（§3.2）。
4. **行 5 的秩 1/2 在网格之外**，且**不得**用其结果论证秩-2 的必要性（§4，`experiment_plan:304`）。
5. **行 3 与 E11 不同口径**（validation vs test）且 checkpoint 来源不同（E14 vs E3 lineage）；
   `r=10` 在各 setting 的相对深度不一致（§5.2）。
6. **行 3 只支持 `shared` 头**：`pooled_lowrank` 头是 `encoder`+`decoder` 双权重，
   不存在单个 `H×720` 矩阵可截断；若某 `l_main` 格解析到非 `shared` 头，脚本记为
   `unsupported_head_type` 而不是静默截断。
7. **单 seed 默认**：行 3 默认 `--seeds 2021`（延续 E11 的单 seed 口径）；扩到 3 seed
   会把 validation 前向次数乘以 3（28 setting × 3 seed × (1 全秩 + n 个秩) 次评测）。
   **2026-09-20 补记（登记口径 vs 工具默认）**：本次**正式运行用的是 3 seed**——流水线第 6 步显式传
   `--seeds 2021,2022,2023`，与本 paper §4.0 的"seeds 2021/2022/2023"协议一致（不是工具默认的单 seed）。
   故 `svd_truncation_table_28.csv` 的行数按 **28 setting × 3 seed** 计，而不是 28 行；
   阶段 5 审校不要把它读成"每 setting 一行"。**成本也随之为默认口径的 3 倍**，且该脚本**无 `--gpus`**
   （只有 `--device`）⇒ **单卡顺序执行**：排期据此把行 3 的估计从 0.2–0.5 h 更正为 **1.5–2.5 h**。
7bis. **行 5 的预注册"上界"目前只有文本形态，阶段 5 必须显式对照（2026-09-20 记）**：
   本 paper §4.6 行 5 的预注册预期含一个**定量上界**——"退化幅度上界由 `1−capture(1/2)`
   （14%–35% / 3%–18% 的支路价值）经 `g²` 折算"。实测（读 `e18_writeback.build_row5`）：产出者**只把该预期
   作为文本记录**（`preregistered_expectation`），并计算实测的 `mean_delta_mse_pct` 与
   `cells_degraded_vs_dense`，**但不计算那个上界、也不做对照**。
   **因此阶段 5 审校必须补上这一步**：把实测退化与预注册上界**显式并列比较**（逐 setting 或按均值），
   或**明确写出为何不逐格对照**。**不得**让预注册停留在"写了但从未被检验"的状态——
   这正是本 paper 对"预注册"的一贯要求。
   **另外刻意不代为实现该公式**：`g²` 折算的具体定义在 paper 里是文字表述，由我把它"实现"成一条算式
   等于**替研究决定一个解释**；这属于研究判断，不属于本单元的工具修正。若后续决定要自动化，
   应先冻结公式定义、再像其它判据一样配正负对照。
8. **行 3 依赖 E14 已完成**：本地**没有任何** E14 产物（`research_runs/` 下无
   `phaseformer_L_e14_main_v1/`），因此本脚本在本地只能做 `--dry-run`/`--verify` 的路径检查，
   真实数值必须在服务器上、E14 stage A 与单次 test 读完成后运行。
9. **新格 gate/lr 的 D-2 溯源**：脚本用 `--learning-rate 1e-3` 且 override 里再写
   `learning_rate=1e-3`，与 E14 `arm_command`（`e14_main_matrix.py:440-441`）一致；
   两者都写只是为了 config.json 留证，不会产生"两次设置"的差异。
10. **行 5 不做对照训练**：minipaper 的该行只注册了 `pooled_lowrank rank∈{1,2}`；
    参照（`l_main`/`l_q1_4`/`l_q1_8`）由 E14 提供，脚本不新训。

---

## 9. Stage 2 静态检查清单（下一阶段逐条执行）

1. `PYTHONDONTWRITEBYTECODE=1 python3 -c "compile(open(p).read(), p, 'exec')"` 对两个脚本均通过。
2. `--help` 退出 0，且列出全部约定 flag（`e18_negative.py`：`--stage/--gpus/--output-root/
   --retries/--poll-seconds/--dry-run/--verify`；`e18_svd_truncation.py`：`--e14-root/--manifest/
   --output-root/--ranks/--evaluation-split/--seeds/--dry-run/--verify`）。
3. `--stage all --dry-run` 的 `counts` = `smooth 42`（`boxcar_s0.5` 21 + `causal_ema_s0.5_a0.08` 21）、
   `rank12 36`（`absolute_rank1` 18 + `absolute_rank2` 18），与 `schedule:167-169` 逐行相符。
4. 78 格 `cell_key` 无重复；每条 `command` 不含 `--evaluate-test`；每格 `status == "new"`。
5. 行 1 的覆盖 = 7 setting × 3 seed × 2 level；行 5 的覆盖 = 6 setting × 3 seed × 2 rank，
   且不含 Electricity-336。
6. `--baseline-manifest` 指向真实 E14 manifest 时，`baseline_report.rejected` 为空，
   且 21 个 `l_main` 基线格全部解析出 `status`/`gate_init`/`learning_rate`。
7. E14 stage A 完成后：`e18_svd_truncation.py --verify` 报告 `problems == []`，
   `planned_settings` = 28，`missing_settings` = `[]`。
8. 截断分析的 `svd_truncation_table_28.csv` 恰好 28 行，`split == val`，`records_test == False`，
   无 NaN 空列（缺失值写空串而非 `nan`）。
9. 抽样复算：对 1 个 setting 手工重跑 `svd_truncate(r=10)` → 与脚本的 `truncated_mse` 一致；
   `full_rank_mse` 与 `l_main` 的 `val_mse` 一致（同一 split、同一 checkpoint、同一聚合口径）。
10. 服务器 `results.csv` 行数 = 该次 `--stage` 的格数；`e18_negative_verify.json` 的
    `resolved` = `checked`。
11. `grep -nE "evaluate[_-]test|'test'|\"test\"" <两个脚本>` 的输出逐条人工确认：
    E18 runner 只应出现在注释/文档串与审计列名里；截断分析不得出现 test loader 构造。

---

## 10. 本机已做的验证（无 torch / 无 E14 产物 / 不读 test）

* 两个脚本 `compile(...)` 通过（未用 `py_compile`）；`--help` 与 `--dry-run` 均退出 0。
* `--stage all --dry-run`：78 格、`counts` 与 §9.3 一致、`cell_key` 唯一、
  所有 `command` 不含 `--evaluate-test`（脚本核对输出）。
* 单元级核对：`build_cells` 的行 1/行 5 覆盖、`smooth_level_key` 对两档给出不同键
  （`boxcar_s0.5` / `causal_ema_s0.5_a0.08`）、`smooth_overrides` 的键集与 §3 表一致。
* `load_baseline_index`：用合成 E14 manifest 验证——带 `--evaluate-test` 的条目被拒绝、
  `_arm_match` 不通过的条目被拒绝、合法 `l_main` 条目被索引并挂到同 setting 的 6 个平滑格上。
* `e18_svd_truncation.build_plan`：用合成 E14 root（24 setting × l_main/l_q1_4/l_q1_8 三格）
  验证 run dir 解析、`comparison_rank` 选择（H=96 → 取秩 12 的 `l_q1_8`；
  H=720 → 取秩 90 的 `l_q1_8`）、`problems` 内容与 checkpoint 缺失标记。
* 列覆盖核对：`PER_RANK_FIELDS`/`TABLE_FIELDS` 与行字典逐项比对，无"声明未写"与"写未声明"。
* 数值口径核对：`relative_gap(0.2104, 0.1629) = 28.91%`、`relative_gap(0.1629, 0.1617) = 0.74%`，
  与 E11 §3.2 记录的 "+29% / +0.7%"(test 口径) 一致，说明 gap 定义未改。
* 本机 **无 torch、无 numpy、无 E14 产物**，因此**没有**执行任何真实前向评测或训练，
  也没有运行 `analysis_*` 的 GPU 路径；真实数值验证须在服务器完成。
