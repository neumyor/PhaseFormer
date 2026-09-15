import unittest

import pandas as pd

from src.dataset.input_component_contrasts import (
    SHAM_ADJUSTED_INTERACTION_RENAMES,
    add_plan_interaction_columns,
    macro_mean_by_setting,
    rename_sham_adjusted_interactions,
)


def build_frame():
    """Merged long table with model M0 ('original') and M1 ('weak'), one setting.

    Values are chosen so that the plan-§8.1 Interaction and the legacy
    sham-adjusted contrast differ by a large, easily detectable amount.
    """
    rows = []
    values = {
        ("original", "none", "full"): (1.00, 1.00),
        ("weak", "none", "full"): (0.95, 0.95),
        ("original", "h1", "minus_A"): (1.32, 1.30),
        ("weak", "h1", "minus_A"): (1.29, 1.27),
        ("original", "h1", "sham"): (1.63, 1.58),
        ("weak", "h1", "sham"): (1.30, 1.28),
    }
    for (model, hypothesis, variant), (mse, mae) in values.items():
        rows.append(
            dict(
                dataset="ETTh1",
                horizon=192,
                seed=2021,
                model=model,
                track="retrain",
                input_hypothesis=hypothesis,
                input_variant=variant,
                test_mse=mse,
                test_mae=mae,
            )
        )
    frame = pd.DataFrame(rows)

    full = (
        frame[(frame.input_hypothesis == "none") & (frame.input_variant == "full")]
        [["dataset", "horizon", "seed", "model", "track", "test_mse", "test_mae"]]
        .rename(columns={"test_mse": "full_mse", "test_mae": "full_mae"})
    )
    out = frame.merge(
        full,
        on=["dataset", "horizon", "seed", "model", "track"],
        how="left",
        validate="many_to_one",
    )
    out["delta_mse"] = out.test_mse - out.full_mse
    out["relative_delta_mse"] = out.test_mse / out.full_mse - 1.0
    out["delta_mae"] = out.test_mae - out.full_mae
    out["relative_delta_mae"] = out.test_mae / out.full_mae - 1.0

    sham = (
        out[out.input_variant == "sham"]
        [["dataset", "horizon", "seed", "model", "track", "input_hypothesis",
          "delta_mse", "relative_delta_mse", "delta_mae", "relative_delta_mae"]]
        .rename(columns={
            "delta_mse": "sham_delta_mse",
            "relative_delta_mse": "sham_relative_delta_mse",
            "delta_mae": "sham_delta_mae",
            "relative_delta_mae": "sham_relative_delta_mae",
        })
    )
    out = out.merge(
        sham,
        on=["dataset", "horizon", "seed", "model", "track", "input_hypothesis"],
        how="left",
        validate="many_to_one",
    )
    out["sham_adjusted_delta_mse"] = out.delta_mse - out.sham_delta_mse
    out["sham_adjusted_relative_mse"] = out.relative_delta_mse - out.sham_relative_delta_mse
    out["sham_adjusted_delta_mae"] = out.delta_mae - out.sham_delta_mae
    out["sham_adjusted_relative_mae"] = out.relative_delta_mae - out.sham_relative_delta_mae

    original = (
        out[out.model == "original"]
        [["dataset", "horizon", "seed", "track", "input_hypothesis", "input_variant",
          "delta_mse", "relative_delta_mse", "delta_mae", "relative_delta_mae",
          "sham_adjusted_delta_mse", "sham_adjusted_relative_mse",
          "sham_adjusted_delta_mae", "sham_adjusted_relative_mae"]]
        .rename(columns={
            "delta_mse": "original_delta_mse",
            "relative_delta_mse": "original_relative_delta_mse",
            "delta_mae": "original_delta_mae",
            "relative_delta_mae": "original_relative_delta_mae",
            "sham_adjusted_delta_mse": "original_adjusted_delta_mse",
            "sham_adjusted_relative_mse": "original_adjusted_relative_mse",
            "sham_adjusted_delta_mae": "original_adjusted_delta_mae",
            "sham_adjusted_relative_mae": "original_adjusted_relative_mae",
        })
    )
    return out.merge(
        original,
        on=["dataset", "horizon", "seed", "track", "input_hypothesis", "input_variant"],
        how="left",
        validate="many_to_one",
    )


def add_legacy_interaction_columns(frame):
    """The pre-fix definition: interaction built from sham-adjusted deltas."""
    out = frame.copy()
    out["interaction_mse_vs_original"] = (
        out.sham_adjusted_delta_mse - out.original_adjusted_delta_mse
    )
    out["interaction_relative_mse_vs_original"] = (
        out.sham_adjusted_relative_mse - out.original_adjusted_relative_mse
    )
    out["interaction_mae_vs_original"] = (
        out.sham_adjusted_delta_mae - out.original_adjusted_delta_mae
    )
    out["interaction_relative_mae_vs_original"] = (
        out.sham_adjusted_relative_mae - out.original_adjusted_relative_mae
    )
    return out


def pipeline_frame():
    """Frame as produced by the fixed summarizer: legacy -> rename -> plan §8.1."""
    legacy = add_legacy_interaction_columns(build_frame())
    renamed = rename_sham_adjusted_interactions(legacy)
    return add_plan_interaction_columns(renamed)


def variant_row(frame, model="weak", hypothesis="h1", variant="minus_A"):
    return frame[
        (frame.model == model)
        & (frame.input_hypothesis == hypothesis)
        & (frame.input_variant == variant)
    ].iloc[0]


class PlanInteractionTest(unittest.TestCase):
    def test_interaction_is_model_minus_original_without_sham_term(self):
        result = pipeline_frame()
        row = variant_row(result)
        delta_m1 = 1.29 / 0.95 - 1.0
        delta_m0 = 1.32 / 1.00 - 1.0
        self.assertAlmostEqual(
            row["interaction_relative_mse_vs_original"], delta_m1 - delta_m0, places=12
        )

    def test_legacy_column_equals_plan_interaction_minus_sham_interaction(self):
        result = pipeline_frame()
        row = variant_row(result)
        delta_m1 = 1.29 / 0.95 - 1.0
        delta_m0 = 1.32 / 1.00 - 1.0
        sham_interaction = (1.30 / 0.95 - 1.0) - (1.63 / 1.00 - 1.0)
        self.assertAlmostEqual(
            row["sham_adjusted_interaction_relative_mse"],
            (delta_m1 - delta_m0) - sham_interaction,
            places=12,
        )
        # ... and is therefore a materially different number (the D0 inflation).
        self.assertGreater(
            abs(
                row["sham_adjusted_interaction_relative_mse"]
                - row["interaction_relative_mse_vs_original"]
            ),
            0.2,
        )

    def test_missing_original_columns_raise_in_formal_mode(self):
        frame = build_frame().drop(columns=["original_relative_delta_mse"])
        with self.assertRaises(KeyError):
            add_plan_interaction_columns(frame)
        relaxed = add_plan_interaction_columns(frame, require=False)
        self.assertTrue(relaxed["interaction_relative_mse_vs_original"].isna().all())

    def test_macro_mean_weights_settings_equally(self):
        # ETTh1 has 3 seeds; ETTh2 a single seed whose value is an outlier.
        frame = pd.DataFrame(
            {
                "dataset": ["ETTh1"] * 3 + ["ETTh2"],
                "horizon": [192] * 4,
                "interaction_relative_mse_vs_original": [0.01, 0.01, 0.01, 0.30],
            }
        )
        # Equal weight per setting: (0.01 + 0.30) / 2, not the row mean 0.0825.
        self.assertAlmostEqual(
            macro_mean_by_setting(frame, "interaction_relative_mse_vs_original"),
            0.155,
            places=12,
        )

    def test_rename_is_idempotent_and_complete(self):
        renamed = rename_sham_adjusted_interactions(add_legacy_interaction_columns(build_frame()))
        for old, new in SHAM_ADJUSTED_INTERACTION_RENAMES:
            self.assertIn(new, renamed.columns)
            self.assertNotIn(old, renamed.columns)
        again = rename_sham_adjusted_interactions(renamed)
        self.assertEqual(list(again.columns), list(renamed.columns))


if __name__ == "__main__":
    unittest.main()
