# 单次 test 读取链路的端到端冒烟：挡下一个会让 §4.5/§4.6 测试数字全空的缺陷

> 执行时间：2026-09-20 03:33–03:40（服务器）｜执行者：本会话
> 触发原因：核对"阶段二还有哪条代码路径**从未真正执行过**"——不是静态检查没做，而是
> 有几条路径**只做过静态检查**。结果查出一条会浪费 24+78 个训练 run 的缺陷。

## 1. 为什么要做这一步

阶段二各步的静态检查、门禁预跑、回填预演都已完成（见
`phase2_gate_sweep.md`、各实验的阶段 2/3 文档）。但静态检查有一个共同盲区：
**它只能证明"命令写得对"，不能证明"这条路径真的跑得通"**。

于是逐一核对哪条路径**从未被执行过**：

| 环节 | 是否真正执行过 | 结论 |
|---|---|---|
| E14 训练（步骤 A） | ✅ 正在跑（411 runs） | — |
| E16 解剖 | 冒烟 + 门已过（`02_03_static_check_smoke.md`） | — |
| E18 训练（平滑/秩两类格） | ✅ 冒烟做过（`e18_negative/02_static_check.md` 第 64 行） | — |
| **E17 训练（冻结子空间投影）** | ❌ **只做过静态检查** | 本轮补做 → §2 **通过** |
| **E17/E18 的单次 test 读取** | ❌ **只做过静态检查** | 本轮补做 → §3 **查出严重缺陷** |

判据很直接：E17 文档自己写明，其训练必须走 `run_top2_direction_retention.py` 的
`--basis` + `weak_residual_projection=frozen_subspace`，且**没有该 override 时
`PhaseFormer.__init__` 会抛错**；而这条路径的 24 个 run 排在**步骤 5**——
前面已有 E14（411 runs）与 E16。它若跑不通，代价极大。

## 2. E17 冻结子空间训练冒烟：**通过**

做法：从 `e17_conditional.py --stage plan` 输出的真实命令里取一格
（`frozen_conditional_direction_1__ETTm2-96-s2021`，最便宜的一格），
只改 `--output-dir` 与 `--max-epochs 1`，并**按 runner 的方式启动**
（`[sys.executable, *command]`，`e17_conditional.py:917-918`）。

```text
{"event": "projection_installed", "rank": 1,
 "basis": ".../projectors/ETTm2_96_Q1COND.npy",
 "sha256": "f45bddb05ace2356272bd64d5ca25a397ff12f17e8a3ade0431e814b50ee125e"}
epochs_completed: 1   val_mse: 0.14325   val_mae: 0.26928
test_mse: ''          test_mae: ''          <-- 训练期绝不读 test ✓
checkpoint written: True
```

即：冻结子空间**确实被装进模型**（`projection_installed`，rank 1，基向量与 sha256 与
§4.5 冻结的投影器一致），训练跑通并写出 checkpoint，且**训练期没有读 test**。

## 3. 单次 test 读取：**查出一个会让测试列全空的缺陷（已修）**

用上面那个真实 1-epoch checkpoint 冒烟 `read_test_generic.py`（步骤 5/6 的 test 读取），
并加一个**负对照**（不传 `basis_file`）验证"校验不通过就不读 test"的保护是否真的生效。

### 3.1 缺陷：worker 从不写它自己被要求写的 marker

首次冒烟结果反常——worker 明明**成功**了，父进程却报失败：

```text
# worker 日志（成功记录）
{"cell": "frozen_conditional_direction_1__ETTm2-96-s2021", "status": "read",
 "test_mse": 0.1927950896215126, "test_mae": 0.28717127738078957,
 "recomputed_val_mse": 0.1432530914049068, "recorded_val_mse": 0.14325090731682102,
 "val_relative_difference": 1.5246591639008888e-05, "test_read_once": true}

# 但父进程写到输出 CSV 的是
status='worker_failed'   test_mse=''   test_mae=''
```

根因：父进程 `dispatch()` 判定成功的条件是
**`marker.is_file() and code == 0`**（`read_test_generic.py:325`），
而 worker 分支**只 `print` 记录、从不写 marker**
（`:350-351`，全文件没有任何一处写 `<marker_dir>/<key>.json`）。
模块自己的 docstring 第 28 行还写着"each freshly read cell leaves a per-cell JSON marker"——
**承诺存在，实现缺失**。

后果（若不发现）：步骤 5 的 24 个 run 与步骤 6 的 78 个 run **训练完**之后，
test 读取会把**每一个** cell 记成 `worker_failed`、test 列留空，
`e17_writeback` / `e18_writeback` 于是产出**测试数字全空的 §4.5/§4.6 表**，
而脚本 **exit 0**，流水线继续往下走。即：整条链"成功"，但两张表没有数字。

**对照可见这不是设计如此**：`e14_read_test.py`（步骤 1 用）的 worker **确实写了**
自己的 per-cell artifact（`write_json(artifact_dir / f"{key}.json", payload)`，
`:1600`/`:1210`），父进程也用同一个键去读（`:1331`）。两个读取器一个有写一个没有，
差别就在这里。

### 3.2 修法

按**文档既定的设计**补齐（不是另造机制）：新增 `--marker-file`，worker 在产出记录处
写出该 marker，dispatcher 把它检查的那个路径传给 worker。

### 3.3 修后复测（两个方向都验）

| case | 期望 | 实测 |
|---|---|---|
| **正向**：`basis_file` 正确 | `status=read`，test 指标非空，且复现校验通过 | ✅ `status='read'`，`test_mse=0.19280`、`test_mae=0.28717`，`val_relative_difference=1.525e-05`（远低于容差） |
| **负向**：不传 `basis_file` | **绝不读 test** | ✅ `status='build_failed'`，`test_mse=''`、`test_mae=''`——test 分裂**没有被读**（模型构建即失败，未走到读数那一步） |

负向对照是刻意设计的：只做正向的冒烟无法证明"校验不通过就不读 test"这条协议保护真的生效。

## 4. 另外两条**已记录但未改动**的观察

1. **`read_test_generic.py` 永远 exit 0**：即使有 cell 被拒绝或 worker 崩溃，
   它只在 `finished` 汇总里列出 `rejected`，进程仍返回 0。
   即"部分失败"不会让流水线停下。**补偿机制**：阶段二新增的第 7 步
   `audit_phase2_outputs.py` 会对此判 FAIL（E17/E18 各有一条"每个 cell 都要有 test 指标"）。
   若真要改成硬失败，应改在**流水线**里（策略属于流水线契约），而不是藏在工具内部——
   但那属于对无人值守链路的改动，收益有限（步骤 5/6 之后紧接第 7 步），故本轮**只记录**。
2. **被读取的 run 必须位于仓库根之下**：`:207` 调
   `checkpoint_path.relative_to(ROOT)`，因此 `/tmp` 下的 run 会让 worker 抛
   `ValueError`。真实运行的 run 全部在 `research_runs/` 下，故不影响流水线；
   但这也说明**冒烟必须把 run 放进仓库内**（本轮据此把冒烟根目录改为 `<REPO>/_smoke_e17`，
   跑完即删除）。这条正是我首次冒烟失败的原因——**是我的冒烟搭错了地方**，不是产品缺陷。

## 5. 本轮我自己的搭台错误（与本会话前几轮同一模式，如实记录）

冒烟脚本本身出错 4 次，每次报错外观都与真实缺陷相似：①用 shell 驱动命令 → 被
`--overrides` 的 JSON 重新分词，得到假的 argparse 错误；②`find` 取 run 目录时多退了一级
（`p.parent.parent`）→ `config.json` 找不到；③断言里比较了**未被写进输出 CSV** 的列
（`recomputed_val_mse` 只在 marker/记录里，落盘的是 `val_relative_difference`）；
④`run_case` 成功返回 `0`，我却写 `if not ok: return 1`——**判反了 sentinel**，
导致正向 case 一通过就退出。

**这已是本会话第 6 次同类自伤**（E17 舍入断言、E18 少建列、E14 文件名、E14 结构误读、
E14 查错文件、本轮 4 项）。纪律不变：**每次冒烟报警都要先分清是"工件坏了"还是"搭台错了"**。
本轮的净结果是 **1 个真缺陷 + 4 个我自己的错**——若不逐项核对，最容易发生的是
把假报警当成真问题去"修"一个没坏的脚本。

## 6. 边界

* 冒烟用的是 **1 epoch** 的 checkpoint，故它验证的是**通路**（能否装投影、能否读写、
  校验是否生效），**不是** §4.5 的数值结论——那些要等 24 个正式 run 跑完后由阶段 5 审校判定；
* 只覆盖 ETTm2-96 一格与两条路径（有/无 basis）。E17 的另一个臂
  （`frozen_independent_direction_1`）走同一段代码、只是基向量文件不同，未单独冒烟；
* 负对照命中的是 `build_failed`（模型构建即失败），因此它**没有**验证
  "模型能构建但 val 复现不过 → 拒绝" 这条更细的路径。该路径由
  `--val-tol` 逻辑承载，正向 case 已证明复现校验会被计算（`val_relative_difference` 非空）；
  若要覆盖它，需要故意用一个"能构建但权重不符"的 checkpoint，本轮未做。

---

## 7. 补做：E14 阶段 B（步骤 1）读取路径的冒烟 —— **通过**

### 7.1 为什么还要补这一条

§1 的核对表里，E14 的 test 读取当时没有被列为"未执行"——因为它有一条 `--dry-run` 链路，
而且我反复跑过它的 dry-run 对账（492 cells 恒等式）。但 dry-run **只走解析器**，
不走"建模型 → 复现校验 → 读 test → 写产物"这条真正的读路径。
其文档 `e14_main/04b_test_read_plan.md` 的结构是"要求 / 继承 / 指纹 / CLI / 输出 schema /
静态检查清单"——**是一份计划，没有任何"已执行"的证据**。

而它是**阶段二的第一步**：它跑不通，整条链在起点就停。故补做。

### 7.2 做法（隔离式，且**不污染真实产物根**）

关键约束：`--output-root` **既是**读取器定位 run 的地方，**也是**它写 `results.csv` 与
`test_read/` 的地方。因此直接对着真实根跑会提前生成阶段 B 的产物。

故：把一个**已完成的真实 run** 复制到仓库内的临时根 `<REPO>/_smoke_e14_read/runs/`，
对**该临时根**运行读取器，跑完即删。实测确认隔离有效——真实根在冒烟前后
**既没有 `results.csv` 也没有 `test_read/`**（只有原有的 manifest / runs / *audit* 文件）。

用 `--cells-file` 把工作**限定到一个 cell**（原因见 §7.4）。

### 7.3 结果：**通过**

```text
{"event": "planned", "cells_in_manifest": 492, "cells_selected": 1, "new_cells": 1}
{"event": "launch", "cell": "a1__Electricity-192-s2021", "gpu": "0", "attempt": 1}
{"event": "done",   "cell": "a1__Electricity-192-s2021", "status": "read"}
{"event": "finished", "cells": 1, "accepted": 1, "problems": 0, "warnings": 0,
 "failed_workers": [], "by_status": {"read": 1}}
```

`results.csv` 的那一行：

| 列 | 值 |
|---|---|
| `status` | `read` |
| `test_mse` | **0.1474873702920466** |
| `test_mae` | **0.2392384302796768** |
| `gate_value` | 0.35719528988447546 |
| `nlinear_mse` | 0.6245221099559465 |

即：读取器**确实** 建了模型、读了 test 分裂一次、并把四个指标（含 NLinear 对照与门值）
落到 `results.csv`，同时写出 `test_read_summary.json` 与 per-cell artifact。
另外启动时的 `fingerprint_check` 报 `constants_equal: true`、`parity_cases: 20`、
`parity_failures: []`——`e14_read_test.py` 与 `e14_main_matrix.py` 对协议常量理解一致。

这条 run 属 `a1` 臂（`gold_combo_reliability_s2`）。顺带确认：**`a1` 确实在被真实训练**
（与"用户裁定 A1 全训 24×3"一致），且其门值 0.357 与 `l_*` 臂的 preset 默认不同。

### 7.4 过程中确立的三条操作事实（供后续复用，避免重复踩坑）

| 事实 | 后果 |
|---|---|
| `--cells-file` 的行格式是 **`arm:dataset:horizon:seed`**（不是日志里的 `arm__Dataset-H-sSEED`） | 用日志 token 会报 `cell token must have 4 ':'-separated fields` |
| 解析器**只搜 `<output-root>/runs`** | 不限定 cell 时，它会把 492 个 cell 全枚举一遍，其中 491 个记 `missing_run` |
| **每个 `missing_run` cell 约耗一个 `--poll-seconds` 周期（15 s）** | 不限定的冒烟**跑不完**（492×15 s ≈ 2 h）。本轮首次尝试因此超时，我手动终止了它——**已确认真实根未被写入**，终止是干净的 |

第 3 条同时说明**阶段 1 正式运行时不会遇到这个问题**：那时 411 个新格全部已有 run、
81 个复用格走登记证据，`missing_run` 应为 0（dry-run 对账也印证：`missing_run` 正是
"尚未训练"的格子数）。

### 7.5 本轮我自己的搭台错误（如实记录）

① 先是按**目录名**去匹配 run（`*electricity_h192_l_q1_8*`），但目录名只编码 **mechanism**、
不含臂名（`l_main`/`l_q1_4`/`l_q1_8` 都叫 `weak_residual`），匹配必失败；
② 用日志 token 当 `--cells-file` 的行；③ 未限定 cell 导致枚举 492 格而超时。
加上 §5 记录的 4 项，本会话同类自伤累计 **9 次**。
结论不变：**每次报警都要先分清是"工件坏了"还是"搭台错了"**——本轮净结果是
**0 个新缺陷**（E14 读路径本身是好的），全部报警都出在我的搭台上。
