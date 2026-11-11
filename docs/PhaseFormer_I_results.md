# PhaseFormer-I 与 B0H+ 对比（2026-11-11）

模型：PhaseFormer-I 为交替双轴 Router 网络（1 层，d=16，k=8，W_p 恒等），直接解码，以最后一个完整周期为参考（结构见 `PhaseFormer_biaxial_stage_results.md`）。

方法：在主表的 7 个数据集、4 个预测长度共 28 个 setting 上用单种子 2021 运行，首轮的周期长度取自 B0H+ 在该 setting 的最优配置；随后由 `scripts/biaxial/tune.py` 围绕与 B0H+ 差距大的 setting 持续调参（随机变异加随机森林代理模型，21:12 至次日 10:00，共 1323 个试验）。**选择依据是测试集（项目声明的 test-set selection），所有试验保存在 `research_runs/biaxial_stage_20261110/tune_trials.jsonl`，结果不是盲测或无偏泛化估计。** B0H+ 的数值来自 `PhaseFormer_VLDBJ_b0hp_best_settings.csv`（同样是单种子、测试集选优）。“持平”定义为测试 MSE 不高于 B0H+ 的 0.3%。

首轮平均比 B0H+ 高 4.0%，持平的 setting 3 个；调参后平均高 1.45%，持平的 setting 12 个（见表末统计）。Electricity 与 Traffic 单次试验需要数小时，且大批量会显存不足，这两个数据集的试验很少（Traffic 与 B0H+ 的差距仍为 3.5% 到 8.4%），Weather 试验也较少。ETTh1-96、ETTh1-192、ETTh1-336、ETTm1-720、ETTm2-192、ETTm2-336 的差距在 0.9% 到 2.9%，未达持平。

| setting | B0H+ test MSE | PhaseFormer-I first run | PhaseFormer-I best | gap to B0H+ | trials | matched | changed settings |
|---|---:|---:|---:|---:|---:|:---:|---|
| ETTh1-96 | 0.3460 | 0.3746 | 0.3529 | +1.98% | 97 | no | k=6, d=12, layers=3, heads=4, dropout=0.2, lr=0.0002, gate_init=0.5, w_init=poly, batch_scale=0.5, max_epochs=50 |
| ETTh1-192 | 0.3740 | 0.4062 | 0.3849 | +2.94% | 128 | no | k=6, d=8, layers=3, heads=2, lr=0.0005, huber_delta=0.1, decoder=d0, gate_init=0.9, w_init=poly, mean_token=True, batch_scale=2.0, max_epochs=50 |
| ETTh1-336 | 0.4106 | 0.4274 | 0.4176 | +1.71% | 412 | no | period=8, k=6, layers=2, heads=2, dropout=0.2, lr=0.002, huber_delta=3.0, gate_init=0.5, w_init=poly |
| ETTh1-720 | 0.4164 | 0.4566 | 0.4128 | -0.87% | 37 | yes | d=48, layers=3, heads=2, dropout=0.2, huber_delta=0.1, anchor=none, gate_init=0.5, w_init=poly, batch_scale=0.5, max_epochs=80 |
| ETTh2-96 | 0.2691 | 0.2825 | 0.2690 | -0.04% | 48 | yes | k=4, routers=16, heads=2, dropout=0.3, huber_delta=0.1, decoder=d0, q=2 |
| ETTh2-192 | 0.3323 | 0.3446 | 0.3333 | +0.29% | 93 | yes | period=8, layers=3, routers=4, lr=0.0005, anchor=x_last, decoder=d0, gate_init=0.05, arch=temporal, w_init=poly, mean_token=True, batch_scale=2.0 |
| ETTh2-336 | 0.3582 | 0.3746 | 0.3577 | -0.14% | 31 | yes | routers=2, dropout=0.05, decoder=d0, gate_init=0.99, batch_scale=4.0, max_epochs=80 |
| ETTh2-720 | 0.3817 | 0.4062 | 0.3847 | +0.76% | 160 | no | period=48, k=6, routers=2, heads=4, dropout=0.3, lr=0.0005, huber_delta=0.1, gate_init=0.05, arch=dual |
| ETTm1-96 | 0.2892 | 0.2910 | 0.2866 | -0.92% | 6 | yes | k=12 |
| ETTm1-192 | 0.3205 | 0.3303 | 0.3212 | +0.24% | 42 | yes | layers=3, routers=2, anchor=none, arch=dual, max_epochs=80 |
| ETTm1-336 | 0.3599 | 0.3612 | 0.3607 | +0.23% | 7 | yes | k=12, max_epochs=80 |
| ETTm1-720 | 0.4037 | 0.4156 | 0.4108 | +1.78% | 43 | no | k=12, routers=16, heads=2, gate_init=0.99, max_epochs=80 |
| ETTm2-96 | 0.1597 | 0.1636 | 0.1594 | -0.20% | 47 | yes | period=48, k=12, routers=16, heads=4, lr=0.0005, huber_delta=0.3, q=2, max_epochs=50 |
| ETTm2-192 | 0.2146 | 0.2215 | 0.2165 | +0.90% | 70 | no | period=48, lr=0.0002, anchor=none |
| ETTm2-336 | 0.2647 | 0.2809 | 0.2669 | +0.85% | 127 | no | period=48, routers=2, dropout=0.05, lr=0.002, huber_delta=0.3, anchor=x_last, decoder=d0, q=2, batch_scale=0.5, max_epochs=80 |
| ETTm2-720 | 0.3447 | 0.3604 | 0.3463 | +0.47% | 83 | no | k=16, routers=2, heads=2, dropout=0.2, anchor=x_last, decoder=d0, mean_token=True, batch_scale=0.5, max_epochs=80 |
| Electricity-96 | 0.1287 | 0.1285 | 0.1285 | -0.14% | 1 | yes | initial |
| Electricity-192 | 0.1437 | 0.1452 | 0.1451 | +0.97% | 2 | no | k=12, mean_token=True |
| Electricity-336 | 0.1628 | 0.1610 | 0.1610 | -1.17% | 1 | yes | initial |
| Electricity-720 | 0.1947 | 0.2010 | 0.2010 | +3.25% | 2 | no | initial |
| Traffic-96 | 0.3551 | 0.3675 | 0.3675 | +3.50% | 1 | no | initial |
| Traffic-192 | 0.3681 | 0.3964 | 0.3964 | +7.69% | 2 | no | initial |
| Traffic-336 | 0.3780 | 0.4095 | 0.4095 | +8.35% | 2 | no | initial |
| Traffic-720 | 0.4230 | 0.4458 | 0.4389 | +3.78% | 5 | no | k=16 |
| Weather-96 | 0.1440 | 0.1498 | 0.1444 | +0.28% | 10 | yes | period=8, routers=2, lr=0.002, decoder=d0, gate_init=0.05, q=2, w_init=poly, batch_scale=4.0 |
| Weather-192 | 0.1848 | 0.1911 | 0.1883 | +1.86% | 9 | no | k=4, heads=2, dropout=0.3, lr=0.002, huber_delta=0.1, decoder=d0, q=2, batch_scale=4.0 |
| Weather-336 | 0.2356 | 0.2419 | 0.2407 | +2.15% | 9 | no | routers=4, decoder=d0, max_epochs=80 |
| Weather-720 | 0.3107 | 0.3114 | 0.3114 | +0.20% | 1 | yes | initial |

28 settings, 12 within +0.3% of B0H+, mean gap +1.45%.
