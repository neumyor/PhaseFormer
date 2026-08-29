# HPTC：单一 PhaseFormer 的相位—轨迹—周期有机整合实验

> 状态：预注册，尚未读取本实验结果。实验编号 `hptc_unified_v1`。

## 1. 已有证据与假设

已有结果给出四条直接约束：

1. A1（RCRF+NLinear）在 ETTh2/ETTm2 等 setting 更稳，说明全 horizon 轨迹不能被周期模型
   整体替换。
2. I0（RCRF+ICPT）在 12 个 H96/H192 setting 中有 8 个双指标优于 A1，尤其是 Weather、
   Electricity；但 ETTh2-H96 MSE 回退约 6.5%，说明周期间结构有效但适用域有限。
3. Rolling TriAxis 的历史首选命中率只有约 30.7%–41.8%，hard routing 和风险单调先验均
   失败，因此历史回测不适合承担离散选专家职责。
4. M3 soft ensemble 相对三个完整模型包络平均只再提高 0.79%，而 hard 版本全部失败；可利用
   信号更像连续误差修正，而不是可靠的完整模型选择。

据此提出假设：**把 NLinear 与 ICPT 的职责正交化，并把 rolling evidence 降级为连续收缩，
可以保留 NLinear 的周期水平/长期轨迹稳定性，同时只在历史支持时注入 ICPT 的周期间形状
修正，从而比并列专家路由更稳。**

## 2. 统一结构

模型只包含一个共享 PhaseFormer 和一个 checkpoint：

1. PhaseFormer 主干输出相位预测，负责周期内相位坐标和跨周期同相位关系。
2. NLinear 输出完整残差轨迹；它独占每个未来 24 步周期的均值/水平。
3. ICPT 只读取逐周期去均值的历史，并输出逐周期去均值的形状。最终修正严格投影到零均值
   子空间，因此不能改变 NLinear 的周期水平。
4. 多截点历史回测估计形状线性演化在相同 lead 上的平均误差与方差，只把 ICPT 形状修正
   连续收缩到 NLinear，不进行 expert argmax。
5. 外层继续使用 RCRF：相位可靠时偏向 PhaseFormer，相位不可靠时偏向上述统一残差预测。

形式上，若 `T` 是 NLinear，`S` 是 ICPT 形状，`Π₀` 表示逐未来周期去均值投影，则

`Y_res = T + β(X) · [S - Π₀(T)]`，且每周期 `mean(S - Π₀(T)) = 0`。

这不是 A1/I0/R0 的预测集成：没有三套 PhaseFormer、没有独立 checkpoint、没有完整模型
路由。NLinear、ICPT、rolling evidence 分别是一个可识别生成过程中的水平、形状和收缩项。

## 3. 参数搜索

固定 ICPT 为历史上更稳的 no-PE decoder、d_model 32、1 层；只搜索机制关键量：

| config | rolling shrinkage | 初始形状比例 β | risk scale |
|---|---:|---:|---:|
| H0 `hptc_fixed_b10` | 否 | 0.10 | — |
| H1 `hptc_rolling_b10` | 是 | 0.10 | 1.0 |
| H2 `hptc_rolling_b25` | 是 | 0.25 | 1.0 |
| H3 `hptc_rolling_b50` | 是 | 0.50 | 1.0 |
| H4 `hptc_rolling_b25_r05` | 是 | 0.25 | 0.5 |

共同设置：ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity，L720、P24、30% train、最多
8 epoch、seed 2021、Huber、best-validation checkpoint，禁止读取 test。第一阶段覆盖 H96。
按相对 A1 的 12 指标宏平均选冠军；同时报告相对 A1/I0/R0 逐指标包络。

H96 晋级 H192 的条件：相对 A1 宏平均比值不高于 0.998、至少 4/6 数据集双指标改善、最差
单指标比值不高于 1.01。若通过，只扩展冻结冠军到六数据集 H192；否则停止并报告失败。

## 4. 待填结果

| config | 相对 A1 宏平均比 | 最差比值 | 双指标改善数 | 相对包络宏平均比 | 决策 |
|---|---:|---:|---:|---:|---|
| H0 | 待填 | 待填 | 待填 | 待填 | 待填 |
| H1 | 待填 | 待填 | 待填 | 待填 | 待填 |
| H2 | 待填 | 待填 | 待填 | 待填 | 待填 |
| H3 | 待填 | 待填 | 待填 | 待填 | 待填 |
| H4 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 5. 必做消融与审计

- H0 对 H1：rolling evidence 是否比固定小修正更稳。
- H1/H2/H3：形状修正强度是否存在稳定区间。
- H2 对 H4：历史风险收缩强弱的作用。
- 校验每个未来周期的修正均值绝对值小于 `1e-6`，以及 flag-on/off 共享相位主干初始化。
- 对冠军相对 A1 回放全部 validation sample×channel，按绝对 MAE 差程序化选 baseline 高误差、
  candidate 退化、candidate 改善各 5 例，并分析 `β`、rolling risk、未来分段误差和输入形态。
- 报告参数量、单模型前向耗时；不得把 M3 ensemble 的计算或结果写成 HPTC 的模型贡献。
