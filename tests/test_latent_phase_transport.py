import unittest

import torch

from src.models.latent_phase_transport import LatentPhaseTransportDecoder


class LatentPhaseTransportDecoderTests(unittest.TestCase):
    def make_decoder(self, p_out=4, max_shift=1, prior_logit=5.0):
        return LatentPhaseTransportDecoder(
            p_out=p_out,
            latent_dim=4,
            hidden=8,
            max_shift=max_shift,
            prior_logit=prior_logit,
        )

    def test_output_shape(self):
        decoder = self.make_decoder(p_out=5)
        z = torch.randn(2, 3, 6, 4)
        output = decoder(z)
        self.assertEqual(output.shape, (2, 3, 6, 5))

    def test_zero_shift_is_exactly_original_linear_predictor(self):
        decoder = self.make_decoder(p_out=4, max_shift=0)
        z = torch.randn(2, 3, 6, 4)
        expected = decoder.value_projection(z)
        actual = decoder(z)
        torch.testing.assert_close(actual, expected)

    def test_transport_is_circularly_equivariant(self):
        decoder = self.make_decoder(p_out=3)
        z = torch.randn(2, 2, 7, 4)
        expected = torch.roll(decoder(z), shifts=2, dims=2)
        actual = decoder(torch.roll(z, shifts=2, dims=2))
        torch.testing.assert_close(actual, expected)

    def test_identity_prior_dominates_initial_transport(self):
        decoder = self.make_decoder(p_out=3, prior_logit=5.0)
        z = torch.randn(1, 2, 5, 4)
        info = decoder.diagnostics(z)
        self.assertTrue(torch.all(info["identity_weight"] > 0.98))
        torch.testing.assert_close(
            info["shift_weights"].sum(dim=-1),
            torch.ones(1, 2, 3, 5),
        )


if __name__ == "__main__":
    unittest.main()
