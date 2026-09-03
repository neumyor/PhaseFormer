# D6：Phase Folding 的结构关系冻结筛查

固定 ETTm1/L720/H192/seed2021、512个均匀 validation origins；不训练、不读取 test。

这轮不再宣称删除一个加性成分，而是测试 PhaseFormer 的 phase folding 是否弱化某类时间关系。三个
确定性扰动均保留最后一个输入点：

|条件|保持|破坏|
|---|---|---|
|cycle-order-reverse|每个 phase 的值集合、每周期内部波形|前29个周期的时间顺序|
|phase-desync|每个 phase 在前29周期的完整值集合|同一周期内不同 phase 的共同演化/相邻关系|
|adjacent-pair-swap|每个周期的值集合、均值和幅度|前29周期的相邻 phase-slot 边|

对 M0/M1/M2 测完整输出，并对 M1/M2 做固定 full phase+gate、仅替换 NLinear branch 的反事实。它是
探索性筛查：只有 M0 对关系扰动近零、增强 NLinear-only 显著受损，才扩展为完整 validation 和匹配 control。
