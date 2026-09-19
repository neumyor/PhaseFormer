# E17 / minipaper §4.5 条件性学习四臂 —— 阶段 1：实验设计

> 状态：**代码已撰写，未做任何运行**（本地无 torch / numpy，且执行契约要求全部训练在远程
> A800 上进行，见 `docs/PhaseFormer_L_execution_schedule.md` §1 与 §5）。
> 本文是六阶段流程的第 1 阶段文档；第 2–6 阶段的产物命名为
> `02_static_check.md` … `06_writeback.md`，与本文同目录。

本单元对应：

| 出处 | 内容 |
|---|---|
| `docs/PhaseFormer_L_minipaper.md` §4.5 | 四臂表 `direct` / 冻结独立-RRR 方向 1 / 冻结条件-RRR 方向 1 / PhaseFormer-L（联合）+ H1 列 |
| `docs/PhaseFormer_L_minipaper.md` §3.3.2 | H1 的定义：训练头输入子空间与"条件性 RRR"（目标 `D_cond`）的距离应小于与"独立 RRR"（目标 `D_ind`）的距离 |
| `docs/PhaseFormer_L_minipaper.md` §3.3.3 | 干预臂里的 `Conditional-RRR-only` / `Independent-RRR-only`；本单元的冻结投影器与这几个臂共用同一对象 |
| `docs/PhaseFormer_L_execution_schedule.md` §2.3 | E17 的臂来源与新训预算：**24 runs** |
| `docs/PhaseFormer_L_execution_schedule.md` §1 D-3 | 范围冻结为 **7 个 test-selected setting** |
| `docs/PhaseFormer_lowrank_checkpoint_information_analysis_plan.md` §11 / 表 5 | H1 的既有证据（72 行），本单元**不重算** |

---

## 1. 范围与预算（不得更改）

Setting 域 = 7 个 test-selected setting × seed {2021, 2022, 2023}：

```text
ETTh2-96  ETTh2-720  ETTm2-96  ETTm2-192  Weather-96  Weather-192  Electricity-336
```

| 臂 | 计数口径 | 来源 | 新训 runs |
|---|---|---|---:|
| `direct` | 7 setting × 3 seed = 21 cell | 复用 E14 `l_main`（`stage_a_manifest.json`） | 0 |
| 冻结独立-RRR 方向 1 | 6 setting × 3 seed = 18 cell | 复用 E8 `keep_direction_1`（`top2_direction_retention_v1/results.csv`） | 3 |
| 冻结独立-RRR 方向 1（Electricity-336） | 1 setting × 3 seed = 3 cell | **本单元新训**（E8 未覆盖 Electricity） | 3 |
| 冻结条件-RRR 方向 1 | 7 setting × 3 seed = 21 cell | **本单元新训**（全新臂） | 21 |
| PhaseFormer-L（联合） | 7 setting × 3 seed = 21 cell | 复用 E14 `l_main`（与 `direct` 同源） | 0 |
| **合计** | | | **24** |

24 = 21（冻结条件）+ 3（Electricity-336 冻结独立），与 §2.3 逐一相符。

---

## 2. 三处必须披露的构造性事实

本节的三条是**设计的真实性质**，不是可选的免责声明；`results.csv` /
`e17_summary.json` 里都有对应的显式字段。

### 2.1 `direct` 与 `PhaseFormer-L（联合）` 在实现上是同一配置

PhaseFormer-L 的修正器就是与相位主干**联合训练的无瓶颈约束线性头**
（`mechanism=weak_residual` + `weak_period_residual_head_type=shared`），
即 minipaper §3.4.1 的主工作点。本仓库没有第二个"联合训练"的对象，因此

```text
direct 列  ≡  PhaseFormer-L（联合）列      （逐 cell 同一 run 目录、同一 config_hash）
```

两列数值相同是**构造使然**，不是两次独立实验。`results.csv` 用
`source` 列区分（`reused_e14_stage_a_manifest`）并在 `e17_summary.json` 的
`disclosures.direct_equals_joint` 中记录该恒等（附逐 cell 一致性自检结果）。
四臂表的判别力因此实际来自**冻结条件 vs 冻结独立 vs 联合**三者，
`direct` 列的作用是锚点/可读性，不能当作独立第四臂来解读。

### 2.2 7 个 setting 由 test-set selection 得到，且复用格继承 Stage-0 冻结超参

* 这 7 个 setting 来自 §4.1 的先导证据（test-exposed），不是盲选集合；本单元沿用它，
  并在 `e17_summary.json` 的 `disclosures.test_selected_settings` 中列出。
* §4.5 表内同时存在两套 (gate, lr) 协议：
  - **冻结臂（新增 grid）** 继承 Stage-0 冻结值（D-2 后半句）；本单元取
    E8 `scripts/run_top2_direction_retention_matrix.py::FROZEN` 表中的取值
    （该表引自 `scripts/run_rank_sweep_multiseed_v4.py::SETTING_TABLE`），
    使同一 setting 的"冻结条件"格与"冻结独立"格在超参上严格配对：

    | setting | gate | lr |
    |---|---:|---:|
    | ETTh2-96 / ETTh2-720 | 0.5 | 0.001 |
    | ETTm2-96 | 0.5 | **0.0003** |
    | ETTm2-192 | 0.2 | 0.001 |
    | Weather-96 | 0.2 | **0.0003** |
    | Weather-192 | 0.5 | 0.001 |

    该取值在本地逐格核对过：E8 `results.csv` 的 6 个 `keep_direction_1` setting
    记录的 `gate_init`/`learning_rate` 与上表完全一致（见 §3.2）。
  - **Electricity-336** 无 Stage-0 冻结记录（E8 未覆盖），因此用 D-2 的 preset 默认
    `gate_init=0.2` / `lr=1e-3`，并逐格标注 `hyperparams_source =
    "d2_preset_default_new_setting"`。
  - 复用格保留其**自身** run 里的 (gate, lr)，其值与上表一致（见 §3.2）；
    `results.csv` 的 `gate_init` / `learning_rate` 列对每一行都写出实际值，
    因此表注可以逐格标注而不用概括。
* 结论：**§4.5 表内不得声明全表同超参**；配对比较只在同一 setting 内成立。
* 与 E14 的接口：若 E14 的 `l_main` **新格**是用 D-2 preset 默认 (0.2, 1e-3) 训练的，
  则 `direct`/`joint` 两列与冻结臂在超参上不完全同源。本 runner 不重训 E14 的格，
  也不改写其记录，只在 manifest 的 `protocol.frozen_hyperparams_source` 与
  `results.csv` 的 `note` 列中如实标注；§4.5 表注须写明这一点。

### 2.3 冻结投影器只在训练 split 上拟合

两个投影器都是 **train split only** 的数据工件（§4.2）。条件臂用到的相位预测，
来自训练好的 E14 `l_main` checkpoint 在 **train loader** 上的一次前向；
该前向不改变任何权重，只是把主干输出当作输入视角的统计量。测试集从未被读取：
概率构建脚本的 audit 中带 `"splits_read": ["train"]`，且 `e15_dimension.load_split`
只把 CSV 解析到**验证边界**（`nrows = border2s[1]`），测试行从未进入内存。

---

## 3. 复用审计规则

### 3.1 `direct` / `PhaseFormer-L（联合）`（E14）

来源：`research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json`。
规则：取 `cell["arm"] == "l_main"` 且 `(dataset, horizon)` ∈ 7 setting 且 `seed` ∈ 3 seed
的 cell，**不重建**其 run 目录，直接沿用 manifest 记录的 `status` / `source` /
`run_dir` / `config_hash`。测试证据按 E14 自己的约定读取：
`source.test_evidence == "inline metrics.csv"` 时读该 run 的 `metrics.csv`，
否则用 `source.test_evidence` 字典中的 `test_mse` / `test_mae` 字段。

E14 的 `_arm_match` 明确排除任何带 `weak_residual_projection` 的 run（冻结子空间臂），
因此 E14 的 `l_main` 索引不可能误纳本单元新训的冻结臂。本 runner 直接 `import`
`scripts/phaseformer_L/e14_main_matrix._arm_match`，不复制其逻辑。

### 3.2 冻结独立-RRR 方向 1（E8，6 setting）

来源：`research_runs/top2_direction_retention_v1/results.csv`，`arm == "keep_direction_1"`。
逐格校验（任一不满足则不进入复用清单并在 manifest 中记录原因）：

```text
dataset/horizon ∈ 6 setting       seed ∈ {2021, 2022, 2023}
gate_init  == FROZEN[setting].gate          （0.5 / 0.2，逐 setting）
learning_rate == FROZEN[setting].lr         （0.001 / 0.0003，逐 setting）
test_read_status ∈ {"read", "reused"}       且 test_mse / test_mae 非空
```

E8 的 6 个 setting 的 `gate_init`/`learning_rate` 已在本地 `results.csv` 上核对为
`(0.5, 0.001) / (0.5, 0.001) / (0.5, 0.0003) / (0.2, 0.001) / (0.2, 0.0003) / (0.5, 0.001)`
——与 `FROZEN` 表逐格一致。

### 3.3 冻结独立-RRR 方向 1（Electricity-336，新训 3 runs）

E8 的 projector builder (`scripts/compute_top2_direction_projectors.py`) 的
`DATASETS` 字典只有 `ETTh2/ETTm2/Weather`，E8 也只在 6 个 setting 上跑过，
所以 Electricity-336 的 `Q1` 必须新造。新造的 `Q1` 用**与 E8 完全相同的 recipe**
（dataset train-split 标准化、`Z = x_win - x_last`、`D = y - x_last`、ridge `1e-6`、
`eigh` 降序、Gram-Schmidt），并由 `e17_conditional_projectors.py` 的
**复现 gate** 提供正确性证据：同一次运行里用同一份代码重算 6 个既有 setting 的
`Q1`，与 `research_runs/top2_direction_retention_v1/projectors/<ds>_<H>_Q1.npy`
比较 `|cos|`，并要求 ≥ 0.999（见 §4.3）。

### 3.4 不可复用的东西（明确排除）

* E1 的 `joint_lowrank_rank_sweep_v1`（单 seed、preset 默认超参）——E8 `reuse_audit.json`
  已判 `rejected`；
* E8 的 `keep_direction_1_2` 臂（方向 1+2，不是方向 1）；
* 任何 `weak_residual_projection_arm` 已有值但**不经本 runner 命名约定**的 run
  （防止把 E8 的臂误认成本单元的臂）。

---

## 4. 投影器：定义、约定与获取方式

### 4.1 `D_ind` 与 `D_cond` 的严格定义

在 dataset train-split 标准化空间（与 E8 相同：`load_split` 的 train 均值/总体 std，
ddof=0，零方差通道 std=1）上，对每个窗口 `x` ∈ R^{720}、其最后一步 `x_last = x[-1]`、
对应未来 `H` 步 `y`：

```text
D_ind  = y - x_last                                （独立目标，与 E8 的方向 1 完全同口径）
D_cond = y - y_phi                                 （条件目标，本单元的新对象）
```

其中 `y_phi` 是**相位主干自身**（RevIN 归一化空间）的预测：

```text
mu, sigma = RevIN(x)                      # PhaseFormer.revin.normalize 返回的 stats
y_phi     = phase_forecast_normalized     # 即 (y_phi_denorm - mu) / sigma 所用的同一张量
```

`y_phi` 通过 `_capture_forward` 记录的 `rec["phase_norm"]` 取得；
`rec["phase_norm"] = (y_phi_denorm - mu) / sigma` 与 `target_norm = (y - mu) / sigma`
的差就是原始尺度上的 `y - y_phi`（除以正数 `sigma`，不改变方向信息），
因此 `D_cond` 的拟合可以在归一化空间直接进行。

### 4.2 条件预测的获取方式（二选一，本单元选 (a)）

| 路线 | 做法 | 权重 | 本单元 |
|---|---|---|---|
| **(a) checkpoint 前向** | 载入 E14 `l_main` 的 best-val checkpoint，在 **train loader** 上做一次 `inference_mode` 前向，读 `phase_norm`、`z`、门值 | 无（未加权） | **采用** |
| (b) 闭式路线 | `scripts/compute_phase_conditional_rrr.py` 的 `weighted_rrr_subspace`，目标为 `target_norm - (1-g)·phase_norm - g·anchor`，权重 `g²` | `g²` | 不采用（见下） |

**为什么选 (a)：**

1. 任务给定的 `D_cond` 定义就是 `y - y_phi`（无门、无 anchor）；路线 (b) 的目标含
   `-(1-g)·phase_norm - g·anchor` 与 `g²` 权重，那是"门固定后支路仍需补的量"，
   与 §4.5 要求的 `y - y_phi` **不是同一个对象**。两者都合法，但混用会让 §4.5 的
   "冻结条件方向"与 H1 的定义脱节。
2. 若 `g = 1`，(b) 退化为 `target_norm - anchor`，即 `D_ind`；这不适用于本单元的
   `l_main` 修正器（gate 初始 0.2，实测门值 ~0.3–0.6）。
3. 本单元用 (a) 得到的是**相位主干真实输出**，不引入闭式近似。

路线 (b) 的旧工程（`compute_phase_conditional_rrr.py`）与 E16 的等价实现仍可作
交叉参考；`e17_summary.json` 记录 `conditional_target_route = "train_split_forward_pass"`。

**为什么是 train-only：** 前向只跑 train loader（`build_loaders(..., splits=("train",))`），
投影器是数据工件、不含任何验证/测试信息；条件目标的模型权重来自 train split 上
训练的 checkpoint（其 best-val 选择是训练协议的一部分，不是 test）。测试集在整个
E17 阶段 A 都不被触摸（§5）。

**一个必须披露的耦合：** `D_cond` 依赖一个已训练的相位主干。同一个 setting 的
`Q_cond` 对三个 seed 是**同一个冻结对象**，因此它取哪个 seed 的 checkpoint 计算
必须固定并记录（`--phase-seed`，默认 2021；核心理由：2021 是 E8/E14 最早完成的
seed，运行成本最低且已完成）。该选择只影响 `D_cond` 的数值，不影响"train-only"性质；
audit 里逐 setting 记录 `phase_checkpoint` 与 `phase_seed`。

### 4.3 两个空间、两条路线（避免口径混淆）

`e17_conditional_projectors.py` 里刻意把两件事分开，因为它们的输入空间不同：

| 路线 | 输入空间 | 用途 | 正确性证据 |
|---|---|---|---|
| `standardized`（E8 recipe） | dataset train-split 标准化后的 `Z = x_win - x_last`（**不含** RevIN 的逐窗 1/σ） | 独立臂的冻结 `Q1` | 与 E8 既有 6 个 `Q1.npy` 的 `|cos|` 复现 gate |
| `revin`（模型路线） | 分支真实输入 `z = (x - x_last)/σ`（无额外权重；与 `compute_phase_conditional_rrr.py` 的"revin 空间"定义一致） | 条件臂的冻结 `Q_cond`（被训练使用），独立性作为诊断 | 相位前向的 `valid_pairs` / `gate` 统计与 `z` 的有限性检查 |

两条路线的 `Z` 只差逐样本正标量 `1/σ`；RRR 的**方向**由 `M_zz` 的**特征向量**给出
（对正标量缩放不变），但若改为对 `M_zz` 加权 `σ²` 就会变成不同的矩阵、不同的方向。
本单元两者都不额外加权，从而：
(a) `standardized` 路线可以逐位复现 E8；
(b) `revin` 路线保持"目标加权对称"——`D_ind` 与 `D_cond` 都在同一个未加权输入度量下比较，
这正是 §4.5 需要在两个**目标**之间做的对比。

### 4.4 投影器文件约定（与 E8 `Q1.npy` 完全一致）

```text
文件名： <output-dir>/<dataset>_<horizon>_Q1.npy          # 冻结独立-RRR 方向 1
         <output-dir>/<dataset>_<horizon>_Q1COND.npy      # 冻结条件-RRR 方向 1
dtype  ： float64 ('<f8')
shape  ： (720, 1)      # 输入侧，720 = --lookback，1 = 只保留方向 1
布局   ： C-order、已正交单位化（Gram-Schmidt）
含义   ： Q Qᵀ 把分支私有输入 (x − x_last)/σ 投影到该方向
消费者 ： scripts/run_top2_direction_retention.py --basis <path>
         （其 _load_basis 断言 ndim==2 且 shape[0]==--lookback）
```

实测既有文件元数据（本地读取
`research_runs/top2_direction_retention_v1/projectors/ETTh2_96_Q1.npy`）：
`{'descr': '<f8', 'fortran_order': False, 'shape': (720, 1)}`，文件大小 5888 字节，
与上表一致。audit JSON 对每个新文件记录 `sha256`、`shape`、`dtype`、
`orthonormality_error`、`projector_idempotence_error`、`|cos|` 复现值。

### 4.5 投影器 CLI

工程的**唯一** `--output-dir` 是 `--output-root`：

```bash
python scripts/phaseformer_L/e17_conditional_projectors.py \
    --output-dir research_runs/phaseformer_L_e17_conditional_v1/projectors \
    --e14-root   research_runs/phaseformer_L_e14_main_v1 \
    --reference-dir research_runs/top2_direction_retention_v1/projectors \
    --dry-run
```

训练 runner 的投影器目录默认即 `--output-root/projectors`，故两者天然对齐；
runner 侧另有 `--projector-dir` / `--projector-audit` 显式覆盖。

`--help` / `--dry-run` 不导入 numpy/torch（惰性导入），可在任意机器上跑。
输出（`--output-dir`）：

```text
<ds>_<H>_Q1.npy             独立-RRR 方向 1（E8 recipe，训练用）
<ds>_<H>_Q1COND.npy         条件-RRR 方向 1（本单元新对象，训练用）
<ds>_<H>_Q1INDREVIN.npy     诊断用：revin 空间的独立方向（不参与训练）
projectors.json             逐 setting 的文件名 / sha256 / 复现 |cos|
projector_audit.json        完整审计（每 setting 一条记录）
reproduction_gate.json      6 个既有 setting 的 |cos| 复现结果与失败清单
```

---

## 5. 训练协议与命令构造

### 5.1 runner 与冻结投影的安装（`--basis` + override）

E14 的 main runner 是 `scripts/search_phaseformer.py`，它的参数表里**没有** `--basis`；
冻结子空间只能通过 `scripts/run_top2_direction_retention.py` 这个 wrapper 安装：

```text
run_top2_direction_retention.py --basis <path> <search_phaseformer 的全部参数>
    → 内部只剥离 --basis / --basis-sha256，其余参数原样交给 search_phaseformer
    → 构造 ProjectedPhaseFormer(original) 子类，PhaseFormer.__init__(configs, projection_basis=basis)
    → head.set_projection_basis(basis)：非 buffer、不改变参数量
```

同时必须传 `--overrides {"weak_residual_projection": "frozen_subspace", ...}`：
`PhaseFormer.__init__` 在装 basis 之前校验 `configs.weak_residual_projection`，
取值为 `"frozen_subspace"` 才允许 `shared` 头；否则抛 `ValueError`。
两处（`--basis` 与 override 键）**都要有**，只给一处会失败。

新训 cell 的 argv（逐字复制 E8 `arm_command` 的 override 结构 + E14 的协议常量）：

```text
scripts/run_top2_direction_retention.py
  --output-dir <output-root>
  --dataset <DS> --horizon <H> --seed <S>
  --stage confirm --lookback 720 --period 24 --max-epochs 30
  --loss huber --percent 100
  --learning-rate <FROZEN_LR[setting]>
  --require-cuda --resume --num-workers 4 --bad-case-limit 0
  --mechanism weak_residual
  --basis <projector-dir>/<DS>_<H>_Q1COND.npy        # 或 _Q1.npy
  --overrides {"learning_rate": <lr>,
               "weak_period_residual_gate_init": <gate>,
               "weak_period_residual_head_type": "shared",
               "weak_residual_projection": "frozen_subspace",
               "weak_residual_projection_arm": "e17_frozen_conditional_direction_1"}
```

`weak_residual_projection_arm` 用**本单元自己的**取值
（`e17_frozen_conditional_direction_1` / `e17_frozen_independent_direction_1`），
理由是：该字段进入 `config_hash` 与 run id，必须能把 E17 的冻结 run 与 E8 的
`keep_direction_1`、E14 的 `l_main` 区分开；E8 的取值 `keep_direction_1` 不能被复用，
否则两条不同投影的 run 会共用同一 run id 前缀。

### 5.2 协议常量（与 E14 逐字对齐）

| 项 | 取值 | 来源 |
|---|---|---|
| interpreter | `/home/yyk/yyk03/miniconda3/envs/time/bin/python` | schedule §5 |
| lookback / period | 720 / 24 | E14 `LOOKBACK` / `PERIOD` |
| stage / epochs | `confirm` / 30 | E14 `--stage confirm`、`MAX_EPOCHS` |
| loss / percent | huber / 100 | E14 `LOSS` / `PERCENT` |
| seeds | 2021 / 2022 / 2023 | E14 `SEEDS` |
| checkpoint | best validation loss（`--resume` 语义） | schedule §5 |
| 新格超参 | 冻结臂用 `FROZEN[setting]`；Electricity-336 用 (0.2, 1e-3) | §2.2 |
| `--require-cuda` / `--resume` | 都传 | 任务要求 + E8/E14 |
| 一卡一 run | `CUDA_VISIBLE_DEVICES=<gpu>`，`gpus` 池轮转 | schedule §4.3 |
| 重试 | `--retries` 默认 1 | E8/E14 默认 |

### 5.3 `--evaluate-test` 的处置（显式 no-op）

本 runner **绝不**向训练进程传 `--evaluate-test`，并在 `dispatch()` 入口做
assert（若任何 cell 的 argv 含该旗标则直接 `SystemExit`）。同时提供：

* `--no-test-read`（默认）：什么都不做，只在 manifest 里写
  `"reads_test": false`；
* `--test-read-plan`（plan/dry-run 阶段可用）：**只打印**"后续单次 test 读取"
  将要执行的目标清单（cell 键 + checkpoint 路径），不执行任何读取。

意义：§4.5 的单次 test 读取是一个**独立阶段**（沿用 E14 stage B /
E8 `read_top2_direction_retention_test.py` 的"每 checkpoint 只读一次"约定），
本 runner 只负责把 checkpoint 冻结好并给出清单。

---

## 6. 输出产物（`--output-root`）

| 文件 | 内容 |
|---|---|
| `stage_a_manifest.json` | 协议块、`reads_test=false`、每 cell 的 `key`/`status`/`source`/`command`、复用审计（E14 + E8）、投影器 sha256、披露字段 |
| `e17_results.csv`（`--results-name` 可改，默认即 `results.csv`） | 每行 = 臂 × setting × seed，含 `source` 列与（如可用）val/test 指标 |
| `e17_summary.json` | 计数、`disclosures`、H1 汇总、`conditional_target_route` |
| `stage_a_summary.json` | 24 个新 run 的完成/失败清单 |
| `projector_audit.json`（由投影器脚本写入 `<output-root>/projectors/`） | 复现 gate 的 `|cos|`、每个投影器的 sha256/shape/dtype/正交误差；runner 通过 `--projector-audit` 链接进来 |
| `_logs/<key>.log` | 每个训练 run 的 stdout/stderr |
| `runs/<run_id>/…` | 训练原始产物（`config.json` / `metrics.csv` / `attempts/*/checkpoints`） |

三个 stage（`--stage plan` / `assemble` / `a`）：`plan` 只写 manifest；
`assemble` 不训练，只把三类来源（E14 manifest、E8 results.csv、新 run 目录）
重新读一遍并重写 `results.csv` + `e17_summary.json`（新增 run、补跑、重算表时用）；
`a` 先训练 24 个新格再装配。

`results.csv` 列（顺序固定）：

```text
arm, dataset, horizon, seed, setting, source, status,
val_mse, val_mae, test_mse, test_mae, test_evidence, gate_init, learning_rate,
basis_file, run_dir, config_hash,
h1_cond_gt_indep_seed_majority, h1_cond_minus_indep_mean, h1_ranks, h1_ranks_supporting,
note
```

`source` 取值域（闭集）：

```text
reused_e14_stage_a_manifest        direct / joint
reused_e8_results_csv              冻结独立（6 setting × 3 seed）
new_trained_e17                    冻结独立（Electricity-336）+ 冻结条件（全部 21）
gap                                声明范围外且未训练（文档化缺口，不静默填值）
```

`test_mse` / `test_mae` 对 `new_trained_e17` 的在阶段 A 一律为空，
`test_evidence` 写明 `pending_single_test_read`；空值不得被后续脚本当作 0 或推断值。

---

## 7. H1 列：复用既有证据，不重算

* 来源：`docs/PhaseFormer_lowrank_checkpoint_information_analysis_plan.md` 表 5
  （"独立目标与条件性目标对齐"），本地核查为 **72 行** = 6 setting × 3 seed × 4 rank，
  **不含 Electricity-336**（§11 已登记）。
* 传入方式：`--h1-evidence <path>`（默认即该 md 文件），runner **只解析不改写**；
  也可给 CSV（表头见 `research_runs/lowrank_checkpoint_information_v1/conditional_rrr_alignment.csv`）。
* 读取约定（minipaper 要求"在 seed 内多数"）：表 5 每行有一个 `支持 H1` 布尔
  （`cond overlap > indep overlap`）。**seed 级判定 = 该 seed 的 4 个 rank 行中
  严格多数（≥3/4）为"是"**；`h1_ranks_supporting` 记录 `n/4`。
  setting 级另给 3 个 seed 的多数计数 `h1_seed_majority_count`（记录而不单独判定，
  避免与 §4.5 的"seed 数"列重复定义）。
* Electricity-336 的 H1 列为 `evidence_missing`（不得留空被当作"否"）；
  若要补，必须另立分析单元（需要 E14 的 Electricity-336 低秩 checkpoint 做同口径
  的子空间重叠），本单元不做。
* 若 `--h1-evidence` 缺失或解析失败：`e17_summary.json` 记
  `h1_evidence_status = "missing"|"unparsable"`，`results.csv` 的 H1 列留空，
  **不阻断训练与表格装配**。

---

## 8. 六阶段执行计划

| 阶段 | 命令 | Gate |
|---|---|---|
| 1 撰写代码 | 本文件 | 设计可追溯到 §4.5 表项（§1–§7） |
| 2 静态检查 | `python3 -c "compile(open(p).read(), p, 'exec')"`；`--help`；`--dry-run` | 两个脚本编译通过；dry-run 的 run 数 = 24（21 + 3）逐条对账 |
| 3 冒烟 | `--stage plan` + 真实 `scripts/run_top2_direction_retention.py --max-epochs 1` 各 1 个（ETTh2-96 与 Electricity-336） | 链路通、`projection_installed` 日志出现、`config.json` 带 `frozen_subspace` |
| 4 正式 | `--stage a --gpus 0,…,7` | 24/24 run 有 `metrics.csv`，无 silent failure |
| 5 审校 | 逐 cell 校验 `source` / 协议字段 / sha256 对账 | 行数 = 4 臂 × 7 setting × 3 seed = 84 行（含重复列），来源 100% 可溯 |
| 6 回填 | 回填 minipaper §4.5 四臂表 + `docs/agent-log.md` | 每个空位有产物路径；未完成项留白 |

---

## 9. 与既有脚本的不一致与歧义（已核对，逐条留痕）

| # | 事实 | 本单元的处理 |
|---|---|---|
| 1 | `scripts/search_phaseformer.py` 的参数表里**没有** `--basis`；冻结子空间只能经 `scripts/run_top2_direction_retention.py` 这个 wrapper 安装 | 新训格一律走 wrapper（argv 里 `[0]` 即该路径）；wrapper 只剥离 `--basis`/`--basis-sha256`，其余参数原样下传，因此 `--percent 100` 等 E14 协议常量可以共存 |
| 2 | E8 的 `FROZEN` 表**不是**统一 0.5/1e-3：`ETTm2-96` 与 `Weather-96` 的 lr 是 **0.0003** | §2.3 的"复用格用 D-2 Stage-0 冻结 (gate, lr)"按字面执行（新建格与同 setting 的 E8 复用格严格配对）；已在 §2.2 给出逐 setting 表，并在 `results.csv` 逐行写出实际值 |
| 3 | E8/Schedule 说的"Stage-0 冻结值"与 E14 §2.2 的"D-2 preset 默认 0.2/1e-3"是两套取值 | 直接冲突时优先 §2.3 的字面要求；Electricity-336 的两臂用 D-2 preset 默认并逐格标注 |
| 4 | E8 的 `config.json` **不记录** `--basis` 的路径与 sha256（`hyperparams._projection_basis` 只在 test-read 脚本里临时构造） | 本单元用不同的 `weak_residual_projection_arm` 取值区分 E17 两条冻结臂，避免不同投影器共用同一 run id；E8 的 `keep_direction_1` 取值**不复用** |
| 5 | E14 的 `_arm_match` 明确排除任何带 `weak_residual_projection` 的 run；E16 的 `build_bases` 也拒绝这类 run | E17 的 run 因此**不会**被 E14/E16 误纳，是预期行为；E17 自己的臂归属直接读 `config["hyperparams"]["weak_residual_projection_arm"]` |
| 6 | 本地 `research_runs/phaseformer_L_e14_main_v1/` 不存在 | 依赖 E14 的解析全部写成了**显式失败 + 明确提示**；本地只用合成 manifest 做过装配链路的自测 |
| 7 | E8 的 `keep_direction_1` 只有 6 个 setting（无 Electricity-336） | 条件臂的 7×3 与独立臂的 3 个新格由此而来；§4.5 表注须逐格标注 6/7 与 7/7 的差别 |
| 8 | H1 既有证据（表 5）只有 6 个 setting、共 72 行，且**无 seed 列** | Electricity-336 的 H1 列写 `evidence_missing`（不推断）；表 5 的行到 seed 的映射按"每 setting 连续 4 行一组"约定，写进 `e17_summary.json.h1_seed_assignment` |
| 9 | 本单元新训的 24 个 run 在阶段 A 没有 test 指标（`--evaluate-test` 被禁止） | `results.csv` 的 test 列留空、`test_evidence = pending_single_test_read`；空值不得被下游当作 0 或推断值 |
| 10 | `direct` 与 `joint` 在实现上是同一对象 | `e17_summary.json.disclosures.direct_equals_joint` + `joint` 行 `note` 双重标注；逐 cell `run_dir` 一致性可由 `results.csv` 直接核验 |

---

## 10. 服务器上仍需验证的事项（本地无法验证）

本地无 torch / numpy，故下列事项**未**经过运行验证：

1. `_capture_forward` 记录的 `phase_norm` 对 `weak_residual + shared` 的 `l_main`
   checkpoint 是否恒为 `denorm_calls[1]`（若统计口径不同，`D_cond` 会取到错误张量）。
   脚本内置的保护是：要求 `records["phase_norm"]` 与 `records["z"]` 非 None、
   `valid_pairs > 0`、二阶矩全部有限；建议在服务器上先对 ETTh2-96 跑一次，
   核对 audit 里的 `n_pairs` 与 `batches` 是否等于 `ceil(train windows / batch)`。
2. `build_loaders` / `build_model` 在服务器 conda `time` 环境下的可用性与
   `time_mark_dim` 注入行为。
3. Ridge `1e-6` 下 `S = Szyᵀ(Szz+εI)⁻¹Szy` 在 Electricity-336（321 通道）上的
   数值条件与 `|cos|` 复现值（预期 ≥ 0.999，若某 setting 低于阈值须人工判读）。
4. `run_top2_direction_retention.py --percent 100` 的端到端接受性（本单元首次把
   该 wrapper 与 `--percent` 组合使用）。
5. E14 `l_main` 是否已在 7 个 setting 上都有 3 seed 的产物
   （`--verify` 会在 manifest 缺格时给出明确失败信息）。
6. 24 个 run 的实际 wall-clock（Electricity-336 约 1717 s/run 量级，3 个 run 顺延）。
