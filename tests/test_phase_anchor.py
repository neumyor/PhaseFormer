import unittest

import torch

from src.models.phase_anchor import PhaseAnchorTransform


class PhaseAnchorTransformTests(unittest.TestCase):
    def test_complete_periods_use_latest_value_of_each_phase(self):
        transform = PhaseAnchorTransform(period_len=4)
        x = torch.arange(8, dtype=torch.float32).view(1, 1, 8)

        phase_series, anchor = transform(x)

        expected_series = torch.tensor(
            [[[[0.0, 4.0], [1.0, 5.0], [2.0, 6.0], [3.0, 7.0]]]]
        )
        torch.testing.assert_close(phase_series, expected_series)
        torch.testing.assert_close(anchor, torch.tensor([[[4.0, 5.0, 6.0, 7.0]]]))
        torch.testing.assert_close(
            transform.center(phase_series, anchor)[..., -1],
            torch.zeros(1, 1, 4),
        )

    def test_incomplete_period_is_filled_from_real_phase_anchors(self):
        transform = PhaseAnchorTransform(period_len=4)
        x = torch.arange(10, dtype=torch.float32).view(1, 1, 10)

        phase_series, anchor = transform(x)

        expected_series = torch.tensor(
            [
                [
                    [
                        [0.0, 4.0, 8.0],
                        [1.0, 5.0, 9.0],
                        [2.0, 6.0, 6.0],
                        [3.0, 7.0, 7.0],
                    ]
                ]
            ]
        )
        torch.testing.assert_close(phase_series, expected_series)
        torch.testing.assert_close(anchor, torch.tensor([[[8.0, 9.0, 6.0, 7.0]]]))
        torch.testing.assert_close(
            transform.center(phase_series, anchor)[..., -1],
            torch.zeros(1, 1, 4),
        )

    def test_coordinate_transform_is_phase_translation_equivariant(self):
        transform = PhaseAnchorTransform(period_len=4)
        x = torch.arange(10, dtype=torch.float32).view(1, 1, 10)
        phase_offset = torch.tensor([10.0, 20.0, 30.0, 40.0])
        shifted = x + phase_offset.repeat(3)[: x.size(-1)].view(1, 1, -1)

        phase_series, anchor = transform(x)
        shifted_series, shifted_anchor = transform(shifted)

        torch.testing.assert_close(
            transform.center(phase_series, anchor),
            transform.center(shifted_series, shifted_anchor),
        )
        displacement = torch.randn(1, 1, 4, 3)
        torch.testing.assert_close(
            transform.restore(displacement, shifted_anchor),
            transform.restore(displacement, anchor)
            + phase_offset.view(1, 1, 4, 1),
        )

    def test_requires_one_complete_period(self):
        transform = PhaseAnchorTransform(period_len=4)
        with self.assertRaisesRegex(ValueError, "one complete input period"):
            transform(torch.zeros(1, 1, 3))


if __name__ == "__main__":
    unittest.main()
