"""Component-level MSE comparison of efficient forecasters.

Every model is trained on the same lookback-720 pipeline and its whole test
split is dumped to disk, so later trend/periodic decompositions are free.
"""
