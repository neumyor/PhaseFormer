# E18 · §4.6 负对照 — 阶段 6 前置：回填工具在**完整与退化输入**下的预演

> 工具：`scripts/phaseformer_L/rehearse_e18_writeback.py`｜预演输出：`/tmp/rehearse_e18/{A,B,C}`
> 状态：**通过**（服务器执行，三个 case 均 exit 0）｜本预演**不训练任何模型**

## 1. 为什么在正式回填前做这件事

`e18_writeback.py` 是阶段二流水线的**最后一步**：它之前已经有 E14 阶段 A（约 411 runs）、
E14 单次 test 读取、E16、E17、E18 训练六步的算力投入。它一旦崩掉，**§4.6 表就没有了**，
而且是在所有昂贵步骤都跑完之后才发现。因此值得用几秒钟的纯 CPU 预演先证明它能活下来。

同时它检验一个静态检查**覆盖不到**的维度：**退化输入**。列名契约（见
`e14_main/05_audit.md` §15）只能证明"列名对得上"，不能证明"取到 None 时会怎样"。
而"取到 None"恰恰是真实运行里最可能的形态——某个 cell 的 test 指标没读到、
某个 setting 不在 E14 的复用集合里。

## 2. 三个 case 与断言

输入的三份 fixture 的**列集全部由生产者自身声明导出**，因此不可能与真实产物漂移：

* E18 结果表 = `e18_negative.RESULTS_FIELDS` **∪ `read_test_generic.py` 盖上去的列**；
* E14 结果表 = `e14_read_test.RESULTS_FIELDS`（`l_main` 作基线）；
* 截断表 = `e18_svd_truncation.TABLE_FIELDS`。

| case | 输入 | 期望 | 实测 |
|---|---|---|---|
| **A** | 全部填满 | 5 行（1–5），verdict 与百分比都算得出 | **exit 0**；row1 `no cell improves both metrics`；row3 `19.0%`；row5 `5.2356%` ✓ |
| **B** | **E14 基线整表为空** | 不崩、退化为空值 | **exit 0**；5 行齐；row5 退化为 `—` ✓ |
| **C** | **E18 行的 test 指标整列为空** | 不崩、退化为空值 | **exit 0**；5 行齐；row5 退化为 `—` ✓ |

另外核对了表的**行结构**：三个 case 的行号都恰为 `['1','2','3','4','5']`，
即 §4.6 的五行的骨架在任何输入下都成立（第 2、4 行是 `KEPT_AS_DASH` 的既有结论，不依赖数据）。

## 3. 预演发现并修掉的一处真实缺陷：`None` 漏进论文表格

**case B/C 首次运行时并没有崩溃，但输出了**：

```text
row 5: addendum='平均 ΔMSE None%'
```

即当百分比算不出来时，Python 的 `None` 被 f-string 直接渲染成字符串 `None`，
**漏进了要写进 minipaper 的表格文字里**。这不是崩溃，因此不会有任何异常提示；
若真实运行中出现（例如 Electricty-336 之外的某个 setting 缺基线），
§4.6 表里就会出现一格 `平均 ΔMSE None%`——**一个会被审稿人看到的明显瑕疵**。

修法（`e18_writeback.py`）：把两处百分比渲染收进一个小函数，`None` 渲染为 `—`，
与表中其它"不可计算"格子的既有表示一致（`KEPT_AS_DASH` 用的就是 `—`）。

修后复测（同一套 case）：

| case | row5 addendum 修前 | 修后 |
|---|---|---|
| A（完整） | `平均 ΔMSE 5.2356%` | `平均 ΔMSE 5.2356%`（**不变**，正常路径未被改动） |
| B（无基线） | `平均 ΔMSE None%` | **`—`** |
| C（空指标） | `平均 ΔMSE None%` | **`—`** |

并且改动后重跑了列名契约门（`check_column_contracts.py --strict`）确认仍 **exit 0**，
即这次修复没有引入新的契约问题。

## 4. 一次**我自己的 fixture 写错**（与 E17 预演同一类教训）

首次运行直接抛错：

```text
ValueError: dict contains fields not in fieldnames: 'test_mae', 'test_mse'
```

原因是我的 fixture 只用了 `e18_negative.RESULTS_FIELDS` 作为列集，而该表记的是
`test_mse_recorded`/`test_mae_recorded`；回填读的 `test_mse`/`test_mae` 是
**`read_test_generic.py` 在合并文件时盖上去的列**。也就是说：

* 这**不是**产品缺陷，而是我的 fixture 少建了列——**同一个"我抽取不到 ≠ 它没生产"的坑**，
  与 `traceability_matrix.md` §5.2 记录的自伤同源；
* 修法不是手写补上两列，而是**直接复用** `check_column_contracts._subscript_store_keys()`
  这个已从源码导出的抽取器，让 fixture 的列集与检查器的模型**共用同一份真相**。

记下它的理由与 §3 相同：预演脚本自身出错时，其报错**外观与真实缺陷无法区分**，
所以每次预演报警都必须先逐项核对是"工件坏了"还是"断言/fixture 写错了"。

## 5. 边界（本预演**没有**证明什么）

* fixture 的**数值**是合成的，因此不验证 §4.6 的结论本身（例如"越平滑越差"是否真的成立）——
  那要等 78 个 run 跑完后由阶段 5 审校按冻结判据判定；
* 它不覆盖 `--e14-results` 指向**错误文件**的情形（那由列名契约门与 §14 的命名契约负责）；
* 它只覆盖了三种输入形态。真实运行若有第四种形态（例如 `stage` 取值缺失），
  仍需按同样方式补一个 case 后再下结论。
