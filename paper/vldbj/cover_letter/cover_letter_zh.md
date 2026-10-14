# 投稿说明信（中文译文）

> 本文件是 [cover_letter.tex](cover_letter.tex) 的中文对照译文，仅供作者内部审阅，不随稿件提交。两者内容如有出入，以英文版为准；修改时请同步更新两个文件。

---

致：*The VLDB Journal* 主编；
特刊“Spatio-Temporal Data Management and Analytics: Machine Learning-based Solutions and Beyond”客座编辑：Christian S. Jensen 教授、Hao Miao 博士、Yan Zhao 教授

尊敬的各位编辑：

我们投递稿件 *“PhaseFormer+: Phase, Level, and Amplitude Tokens for Efficient Spatio-Temporal Forecasting”*（作者：Yiming Niu、Jinliang Deng、Yongxin Tong），申请收录于特刊 *Spatio-Temporal Data Management and Analytics: Machine Learning-based Solutions and Beyond*。

**会议版本披露。** 根据期刊投稿规则，我们在此声明：本稿是我们已经过同行评审的会议论文 *“PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting”*（Y. Niu、J. Deng、Y. Tong，International Conference on Learning Representations (ICLR), 2026）的扩展版本。稿件在三处引用了该会议论文：标题脚注、第 1 节、Declarations；第 1 节概述了新增贡献；下文逐节列出哪些部分为新增、哪些沿用会议版。

**与特刊的相关性。** 本期特刊征集面向时空数据的高效、可扩展方法，预测是其主题之一。PhaseFormer+ 用一个仅几千参数的共享模型预测传感器网络中的每一路传感器流（Traffic 上相位通路约 1.2K，水平与振幅通路至多再加约 2.1K），部署成本随传感器数量线性增长，各传感器可并行处理。其振幅 token 所承载的水平–振幅耦合在七个基准中的五个上符号稳定，并在两个传感器网络上最强，即 Traffic（862 个道路传感器）和 Electricity（321 个电表）（第 3.3 节）。

**新增内容概要。** 会议版提出了相位 token，证明其在各周期被同样重塑时保持在固定子空间中，并提出了基于相位的路由 Transformer。期刊版围绕一个问题重写全文——预测周期序列是否需要当前模型的复杂度？——并以三类 token 给出回答。新增内容如下。

1. **token 需要表征什么（第 3.1、3.3 节；全部为新增）。** 周期序列的一个窗口承载三类信息：周期的形状、水平与振幅，三者构成同一个低秩结构，即水平项加上按振幅缩放的形状。新的能量分解表明，它在全部七个基准上捕获归一化窗口 78%–93% 的能量；与模型无关的测量显示，其中五个基准存在符号稳定的水平–振幅耦合，远高于保留漂移、去掉对齐的参照。

2. **新模型：水平与振幅 token 及其通路（第 4.1、4.3–4.6 节；全部为新增）。** PhaseFormer+ 新增水平 token（锚定的周期均值），由秩 3 的水平通路映射到固定的偏移、倾斜、曲率轮廓上；以及由水平计算得到的振幅 token，通过中心化调制缩放周期形状，并可证明保持每个未来周期均值不变；两个门控融合各路预测。水平与振幅通路按 setting 新增 119 至 679 个参数，另加每通道两个或共享的两个融合门；新增计算量在输入与输出长度上保持线性。

3. **三类 token 的统一低维理论（第 5 节，完整证明见新增附录；全部为新增）。** 分三步：必要性、形式、充分性。对任意参数取值，相位通路把预测水平与周期内振幅限制在两个可由权重算出的包络内，包络之外存在任何额外深度都无法消除的误差下界；在标准季节模型下，包络之外缺失的是固定 horizon 轮廓上的少数系数（低秩水平定理）和一个由水平驱动的因子（振幅包络定理、中心化调制命题）；这些形式的秩 1、宽度 1 通路即已足够。第 5 节新增三个定理、两个命题，附录另有三个引理和两个推论。

4. **新实验（第 6.2、6.5、6.6 节；全部为新增）。** PhaseFormer+ 与八个基线相比，在七个基准中的六个上平均 MSE 最优，并在全部七个上位列前二（第 6.2 节）。在统一协议下的 token 消融逐个加入 token，覆盖全部 28 个“数据集 × 预测长度”setting、三个 seed：PhaseFormer+ 在 7 个数据集均值中的 5 个上改进，最大 3.3%，最大的四个增益恰好落在包络收紧的四个数据集上（第 6.5 节）。在 84 个训练好的模型上，预测水平从未越出包络；水平包络在水平 token 增益最大的三个数据集上收紧；周期均值不变性严格成立（最大偏差 8.9×10⁻¹⁶）；合成乘性序列上的增益随耦合增大；三个固定轮廓与学习解码器相差在 0.13% 以内（第 6.6 节）。

5. **扩展的相关工作与结论（第 2、7 节）。** 相关工作按分解、token 与归一化方法分别覆盖了三类信息中的哪些重新组织，并讨论传感器网络规模的预测；结论新增局限性讨论。

**与会议版的重叠。** 以下部分沿用会议版，仅做编辑性修改：第 3.2 节（Insight 1、2 以及含 Theorem 1 的稳定性分析）、第 4.2 节（相位通路，按第 5 节所依赖的细节表述）、第 6 节中的会议版实验（与八个基线的对比、效率、router 与 routing 消融、相位长度敏感性、案例研究），以及会议版补充材料（含稳定性定理的证明），作为 electronic supplementary material 提交。摘要、引言、相关工作和结论均为重写；整体为新增的有：第 3.1、3.3 节，第 4.3–4.6 节及第 4.1 节的 token 定义，第 5 节，第 6.5、6.6 节，以及三个附录（证明、扩展结果、per-setting 配置）。新增内容因此超过期刊要求的 30%。

**可获取性。** 所有数据集均为公开基准。PhaseFormer+ 的代码及全部实验脚本见 https://github.com/neumyor/PhaseFormer_TSL 。

本稿未在其他地方发表，也未在其他期刊审稿中。全体作者均已同意投稿。

感谢您审阅本稿，期待您的回复。

此致
敬礼

Jinliang Deng
复杂关键软件环境全国重点实验室，北京航空航天大学
南方科技大学可信自主系统研究院
jinliangdeng9588@gmail.com
代表全体作者：Yiming Niu、Jinliang Deng、Yongxin Tong
