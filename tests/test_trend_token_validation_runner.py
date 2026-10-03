from types import SimpleNamespace

from scripts.run_trend_token_validation import build_jobs


def test_controlled_arm_filter_keeps_only_requested_arms():
    args = SimpleNamespace(
        arms=[
            "phase_only",
            "trend_token_direct_no_interaction",
            "trend_token_direct",
        ],
        trend_dims=[32],
        trend_kernels=[5],
        trend_gates=[0.2],
        profile_counts=[1],
        datasets=["ETTh1"],
        horizons=[96],
        seeds=[2021],
        lookback=720,
        period=24,
        epochs=1,
        capacity="base",
        batch_size=0,
        percent=100,
        loss="huber",
        num_workers=0,
        max_eval_samples=0,
        output_dir="research_runs/test_runner",
    )
    jobs = build_jobs(args)
    assert [job["payload"]["arm"] for job in jobs] == [
        "phase_only",
        "trend_token_direct_no_interaction",
        "trend_token_direct",
    ]
