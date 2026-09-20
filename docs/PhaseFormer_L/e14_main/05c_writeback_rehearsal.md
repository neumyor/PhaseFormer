# E14 · §4.2 主表 — 阶段 6 前置：回填工具在**完整与退化输入**下的预演

> 工具：`scripts/phaseformer_L/rehearse_e14_writeback.py`｜预演输出：`/tmp/rehearse_e14/{A,B,C,D,E}`
> 状态：**通过**（服务器执行，五个 case 均 exit 0）｜本预演**不训练任何模型**
> 更新：**2026-09-20 追加 case D/E**（门值回退与缺 `dataset` 列，§4b）——首版只有 A/B/C，
> 未覆盖 §17 那类缺陷的形态；追加的触发器见 §5 末条。

## 1. 为什么这一份最值得做

`e14_writeback.py` 产出的是**论文的中心产物**：§4.2 主表（24 个 setting + 4 个 Traffic 附录格子）。
它同时是这套脚本里 join 最多、逻辑最重的一个：清单（manifest）、带 test 的结果表、
Golden 参照、E19 电平统计、E14 参数量表五路输入，再按冻结阈值判定主张 A–D
与两个必答块。它一旦崩掉，丢的不是一列而**是整张主表**。

用**真实** manifest（492 cells）、**真实** Golden 表（28 setting）、**真实** E19 电平统计；
只有结果行与参数量行是合成的，且其列名取自生产者自身声明，故不会与真实产物漂移。

## 2. 三个 case 与结果（A/B/C 为首版；D/E 见 §4b）

| case | 输入 | 实测 |
|---|---|---|
| **A** | 全部填满 | **exit 0**；`main_table` 28 行（core 24 + appendix 4）；`variant_table` 6 行；`provenance_note` 无空值；claims.json 含 A/B/C/D + 两个必答块；stdout 汇总行含 A–D 判定与 `must_answer_*` |
| **B** | **结果表无 test 指标** | 修前 **exit 1 崩溃两次**（见 §3）；修后 **exit 0**，退化为主表仍 28 行、`variant_table` 0 行 |
| **C** | **无参数量表** | **exit 0**，正常退化（主表与 variant 均完整） |

结构断言按**实际设计**核对：`main_table.csv` 是**每个 Golden setting 一行（28）**，
其中 Traffic 的 4 行由 `is_traffic_appendix=True` 标记——**这个标记列**（而不是另拆一个文件）
才是"24 主 + 4 附录"的分界；`variant_table.csv` 则是**逐臂汇总（6 臂）**。

## 3. 预演发现并修掉的**两处真实缺陷**（同一类：局部缺失 → 整表全丢）

### 3.1 `round(pct_change(...), 4)` 未做 None 保护 → `TypeError`

```text
File ".../e14_writeback.py", line 350, in main
    round(pct_change(entry.get("l_main_mse"), FITS_MSE[(dataset, horizon)]), 4)
TypeError: type NoneType does not define __round__ method
```

`pct_change` 在指标缺失时返回 `None`，而 `round(None, 4)` 抛异常。
**关键在于同一文件里姊妹比较是有保护的**（`phaseformer_l_vs_golden_pct` 写作
`None if ... is None else round(...)`）——所以这是**一处不一致**，不是风格选择。
已按同一风格补上保护。

### 3.2 `variant_rows[0].keys()` 未防空 → `IndexError`

修掉 3.1 后 case B 又在**下一处**炸：

```text
File ".../e14_writeback.py", line 706, in main
    writer = csv.DictWriter(handle, fieldnames=list(variant_rows[0].keys()))
IndexError: list index out of range
```

当没有任何臂有指标时 `variant_rows` 为空，直接取 `[0]` 越界。
同样地，**相邻的主表写入已经有这个保护**（`fieldnames=list(rows[0].keys()) if rows else [...]`），
故按同一风格补上。

**为什么这两处值得修（而不是"真实运行不会发生"）**：真实运行里第 1 步保证所有 cell 都有
test 指标，所以它们在正常路径上不会触发。但触发条件并不苛刻——**只要有一个 setting 的指标缺失**
（例如第 1 步有格子未解析、或某个复用格的证据丢失），当前代码就会**整张 §4.2 主表丢失**，
而不是把那一格渲染成不可计算。这与本套件其它工件的既定约定不一致：minipaper 的表格本就用 `—`
表示算不出来的格子。

### 3.3 修后**正常路径未被改动**（已实测）

case A 的 28 行**全部**仍算出 FITS 差值（`phaseformer_l_vs_fits_pct` 非空 28/28，
如 `ETTh1-96 = -1.2427`、`ETTh1-720 = 4.0229`），即保护只在"确实算不出来"时生效；
`claims.json` 结构不变，`must_answer_a.reaches_fits_count = 0` 正常产出。
另外改动后重跑列名契约门，仍 **exit 0**。

## 4. 两次**我自己的**断言错误（必须与真实缺陷分开记）

预演首轮还报了另外两处"失败"，逐项核对后确认**都是我写错了**，不是产品缺陷：

| 我的断言 | 实际设计 | 结论 |
|---|---|---|
| `main_table` 应 24 行、`variant_table` 应 4 行 | 主表 28 行（含 `is_traffic_appendix` 标记的 4 行），variant 是**逐臂汇总 6 行** | **我误读了设计**，已改为按标记列核对 24+4 |
| 在 `claims.json` 里找 `claim_A_either_metric`/`claim_B` 等 | 这些扁平化判定在 **stdout 的 `{"event":"finished"}` 行**；`claims.json` 的顶层是 `A/B/C/D/must_answer_a/must_answer_b` | **我查错了文件**，已改为分别核对 |

另一个纯属笔误：首轮直接 `FileNotFoundError`，因为我在预演里把 Golden 文件名写成了
`PhaseFormer_golden_standard.md`，而仓库与服务器上的规范名一直是
**`PhaseFormer_gold_standard.md`**（`MANAGE_RULES.md`、minipaper、流水线用的都是后者）。

**这是同一教训的第三次出现**（E17 的舍入断言、E18 的少建列、本次的文件名）：预演脚本
自身出错时，**报错外观与真实缺陷无法区分**，所以每次报警都必须先分清是"工件坏了"
还是"断言/fixture/常量写错了"。本轮 5 个报警里 **2 个是真的、3 个是我的**——
若不逐项核对，既可能去"修"没坏的东西，也可能因为"工具跑通了"而放过真缺陷。

## 4b. 2026-09-20 追加的两个 case：D（门值回退）与 E（缺 dataset 列）

§17 的门列缺陷（回退路径把数据集合并掉了）**在本预演的首版里没有被任何 case 覆盖**——
A/B/C 三个 case 都不含"某臂在某数据集上缺 `results.csv` 门值"这一形态。这两个 case 是针对该形态
**事后补的**，并各自用**旧实现**做过反向对照：

| case | 输入 | 正确行为 | 反向对照（指向**旧**实现） |
|---|---|---|---|
| **D** | `ETTh1`/`Weather` 两数据集的 `results.csv` **无门值**（其余有） | 这两组必须回退到**本格自己**的 checkpoint 门值（夹具把每数据集的先验写成 `0.30 + 0.05·i`，故"回退到别的数据集"会立刻算错） | **`wrong: 24`**，且报出 `the checkpoint fallback is not reading THIS cell's own gate` ✓ |
| **E** | 参数量表**没有 `dataset` 列** | 必须**不报**门值（宁可缺，不借别的数据集的值） | **`wrong: 8`** ✓ |

**两处 fixture 自身的错误（与 §4 同类，故记在这里）**：

1. case D 首版**对每个臂都断言了回退门值**，但 `phase_only`/`l_rcrf`/`a1` **没有门参数**，
   于是它要求一个模型不可能有的数，崩在 `ValueError: could not convert string to float: ''`；
   改为只对三个带门臂断言。
2. case E 首版用 `is not None` 判空，而 **CSV 空单元格读回来是 `""` 而非 `None`**，对正确产出者误报两条 FAIL；
   改用 `str(got).strip() != ""`。这一条尤其值得记：它**正是本预演存在的理由所要抓的那类错误**，
   却发生在预演自己的断言里。

**检测力的标定方式**：把**新夹具**指向**旧实现**（隔离副本 `/tmp/prefix_rehearse`），
D/E 分别报 `wrong: 24` 与 `wrong: 8` 并各带一条 "not reading THIS cell's own gate"；
指向新实现则 **PASS**。即这两个 case 是**检测器**（对旧实现失败），不是对现有行为的描述。

## 5. 边界

* 合成数值无意义，故本预演**不验证** §4.2 的结论本身（主张 A–D 的真假、
  与 Golden/FITS 的比较）——那要等 492 个 cell 的 test 读取完成后由阶段 5 审校按冻结阈值判定；
* 它不覆盖 `--spect`/其它 CLI 组合，也不覆盖写盘失败（磁盘满、权限）这类环境故障；
* case 数量现为**五种**输入形态（A/B/C/D/E）；真实运行若出现第六种形态，需按同样方式补 case 后再下结论；
* **本预演不是充分条件**：case D/E 是在缺陷**已被独立发现之后**才补上的，因此它们证明的是
  "该缺陷若重现会被抓住"，**不**证明"现有 case 集已覆盖所有同类形态"。补 case 的触发器是
  §17 那次 checkpoint 直读取证，而不是预演自己报的警——这一点必须写明，否则会把
  检测器的事后补强误读成"预演本来就能发现它"。
