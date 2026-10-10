# 投稿说明信（中文译文）

> 本文件是 [cover_letter.tex](cover_letter.tex) 的中文对照译文，仅供作者内部审阅，不随稿件提交。两者内容如有出入，以英文版为准；修改时请同步更新两个文件。

---

致：*The VLDB Journal* 主编；
特刊“Spatio-Temporal Data Management and Analytics: Machine Learning-based Solutions and Beyond”客座编辑：Christian S. Jensen 教授、Hao Miao 博士、Yan Zhao 教授

尊敬的各位编辑：

我们投递稿件 *“PhaseFormer: Efficient Phase-Based Forecasting for Spatio-Temporal Data Beyond Stable Cycles”*（作者：Yiming Niu、Jinliang Deng、Yongxin Tong），申请收录于特刊 *Spatio-Temporal Data Management and Analytics: Machine Learning-based Solutions and Beyond*。

**会议版本披露。** 根据期刊投稿规则，我们在此声明：本稿是我们已经过同行评审的会议论文的扩展版本。

> Y. Niu, J. Deng, Y. Tong. *PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting.* International Conference on Learning Representations (ICLR), 2026.

稿件在三处引用了该会议论文：标题脚注、第1节、Declarations。第1节最后一段说明了哪些部分沿用会议版、哪些部分为新增。提交的 PDF 中，相对会议版新增或改动的段落一律以蓝色印刷，审稿人可以直接核对。以下说明供审稿人和负责本稿的主编参阅。

**与特刊的相关性。** 本期特刊征集面向时空数据的高效、可扩展方法，预测以及流数据和边缘计算场景都在征稿主题之列。PhaseFormer 用一个约一千参数的共享模型预测传感器网络中的每一路传感器流，计算成本随传感器数量线性增长，各传感器可以并行处理。扩展部分处理的是大规模传感器网络特有的非平稳性。我们的数据分析表明：日周期的水平与振幅之间的耦合在全部七个基准数据集中都存在，在两个传感器网络上最强，即 Traffic（862 个道路传感器）和 Electricity（321 个电表）。

**新增内容概要。** 会议版证明了 phase token 在周期形状变化下保持稳定，并提出了 PhaseFormer 架构。它自述的局限是假设周期局部稳定，并把非平稳性与复杂漂移列为未来工作。期刊版正是针对这一局限，新增内容如下。

1. **新的问题表述（第1、3节）。** 我们指出会议版稳定性定理覆盖哪些变化：沿折叠序列相位轴的变化。它不覆盖沿周期轴的变化，即周期水平的漂移，以及随水平变化的周期振幅（乘性季节性）。

2. **新理论（第5节、附录B；全部为新增）。** 会议版只有一个定理，期刊版新增四个定理，并给出完整证明：
   - **能力定理**：PhaseFormer 的路由层能精确传递共同的水平平移，并能逼近季节随机游走模型的贝叶斯预测；
   - **水平包络定理**：对任意参数取值，预测水平都被限制在一个由最终层归一化决定的区间内，该区间可以从训练好的权重中读出。定理同时给出显式的误差下界，并刻画了水平与形状之间的竞争；
   - **低秩互补定理**：phase 路径跟随不了的水平只沿少数几个固定的 horizon 轮廓起作用，因此一个秩受限的线性分支就能承载它；
   - **振幅定理**：包括 phase 路径的振幅包络、振幅需求的乘性结构、中心化调制下每个未来周期均值的严格不变性，以及该调制的表达能力。

3. **新模型（第6节；全部为新增）。** PhaseFormer-LA 保持会议版架构不变，增加了三个组件，每个组件都由一个新定理推出：
   - 一个读取周期均值、输出到固定 horizon 轮廓的秩3水平分支；
   - 一个水平/形状融合；
   - 一个中心化振幅调制。

   在所有 setting 下，扩展部分增加 119 至 367 个参数，与预测长度无关；另加每个通道两个融合门，或全局共享的两个融合门。我们给出了它的精确嵌套性质、保持不变的理论保证以及复杂度。

4. **新实验（第8节及第7节部分内容、附录C；全部为新增）**：
   - PhaseFormer-LA 在全部 28 个数据集×预测长度 setting 上的三 seed 结果，加入主对比表；
   - 七个基准数据集的水平–振幅耦合分析，以数据为中心；
   - 从 PhaseFormer 到 PhaseFormer-LA 的逐级消融；
   - 在训练好的 checkpoint 上实测两个包络和水平交接；
   - 在合成乘性序列与真实数据上的机制实验；
   - 水平分支的容量研究。

5. **扩展的相关工作与讨论（第2、9节）。** 新增三方面讨论：面向非平稳预测的归一化方法；经典与深度季节分解（含乘性季节性）；时空预测模型。另新增局限性讨论。

**与会议版的重叠。** 以下部分沿用会议版，仅做编辑性修改：
- 第3节（动机）和第4节（方法）；
- 第7节中的会议版实验，包括与八个基线的主对比、效率对比、router 与 routing 消融、相位长度敏感性和案例研究；
- 附录A，即会议版补充材料，含会议版定理的证明。

摘要、引言、相关工作和结论经过修订和扩写。按正文（第1–9节）词数计，标记为新增或改动的段落约占正文的一半；第5、6、8节以及附录B、C整体为新增。新增内容因此超过期刊要求的 30%。

**可获取性。** 所有数据集均为公开基准。PhaseFormer 与 PhaseFormer-LA 的代码及全部实验脚本见 https://github.com/neumyor/PhaseFormer_TSL 。

本稿未在其他地方发表，也未在其他期刊审稿中。全体作者均已同意投稿。

感谢您审阅本稿，期待您的回复。

此致
敬礼

Jinliang Deng
复杂关键软件环境全国重点实验室，北京航空航天大学
南方科技大学可信自主系统研究院
jinliangdeng9588@gmail.com
代表全体作者：Yiming Niu、Jinliang Deng、Yongxin Tong
