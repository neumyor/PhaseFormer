import unittest

import torch

from src.models.phase_transport import CircularPhaseTransportDecoder


class CircularPhaseTransportDecoderTests(unittest.TestCase):
    def make_decoder(self, p_out=4):
        return CircularPhaseTransportDecoder(
            p_out=p_out,
            latent_dim=4,
            hidden=8,
            memory_size=3,
            max_shift=1,
            prior_logit=3.0,
        )

    def test_output_shape(self):
        decoder = self.make_decoder(p_out=5)
        z = torch.randn(2, 3, 6, 4)
        history = torch.randn(2, 3, 6, 8)
        output = decoder(z, history)
        self.assertEqual(output.shape, (2, 3, 6, 5))

    def test_initial_prior_preserves_constant_period_state(self):
        decoder = self.make_decoder(p_out=4)
        profile = torch.full((6,), 1.25)
        history = profile.view(1, 1, 6, 1).expand(1, 1, 6, 8).clone()
        z = torch.zeros(1, 1, 6, 4)
        output = decoder(z, history)
        expected = profile.view(1, 1, 6, 1).expand_as(output)
        torch.testing.assert_close(output, expected)

    def test_transport_keeps_shape_and_level_paths_separated(self):
        decoder = self.make_decoder(p_out=4)
        profile = torch.tensor([0.5, 1.0, -0.5, 2.0, 1.5, -1.0])
        history = profile.view(1, 1, 6, 1).expand(1, 1, 6, 8).clone()
        z = torch.zeros(1, 1, 6, 4)
        output = decoder(z, history)
        expected_level = profile.mean().expand(1, 1, 4)
        torch.testing.assert_close(output.mean(dim=2), expected_level)

    def test_transport_is_circularly_equivariant_for_fixed_latents(self):
        decoder = self.make_decoder(p_out=3)
        history = torch.randn(2, 2, 7, 6)
        z = torch.randn(2, 2, 7, 4)
        expected = torch.roll(decoder(z, history), shifts=2, dims=2)
        actual = decoder(
            torch.roll(z, shifts=2, dims=2),
            torch.roll(history, shifts=2, dims=2),
        )
        torch.testing.assert_close(actual, expected)

    def test_diagnostics_are_normalized_and_interpretable(self):
        decoder = self.make_decoder(p_out=3)
        z = torch.randn(1, 2, 5, 4)
        history = torch.randn(1, 2, 5, 6)
        info = decoder.diagnostics(z, history)
        torch.testing.assert_close(
            info["memory_weights"].sum(dim=-1),
            torch.ones(1, 2, 3),
        )
        torch.testing.assert_close(
            info["shift_weights"].sum(dim=-1),
            torch.ones(1, 2, 3, 3),
        )


if __name__ == "__main__":
    unittest.main()
