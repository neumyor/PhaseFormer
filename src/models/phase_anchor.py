import torch
import torch.nn as nn


class PhaseAnchorTransform(nn.Module):
    """Express each phase trajectory relative to its latest real observation.

    The transform contains no trainable parameters. For incomplete final
    periods, missing phase slots are filled with their anchors, so their
    centered value is zero instead of a circularly padded observation.
    """

    def __init__(self, period_len: int):
        super().__init__()
        if period_len < 1:
            raise ValueError("period_len must be positive")
        self.period_len = int(period_len)

    def forward(self, x):
        """Return phase trajectories and anchors from ``x`` shaped (B,C,T)."""
        if x.ndim != 3:
            raise ValueError("x must be a rank-3 tensor shaped (B, C, T)")

        seq_len = x.size(-1)
        if seq_len < self.period_len:
            raise ValueError(
                "phase anchoring requires at least one complete input period"
            )

        phase = torch.arange(self.period_len, device=x.device)
        steps_back = torch.remainder(seq_len - 1 - phase, self.period_len)
        last_indices = seq_len - 1 - steps_back
        anchor = x.index_select(dim=-1, index=last_indices)

        pad_len = (-seq_len) % self.period_len
        if pad_len:
            first_missing_phase = self.period_len - pad_len
            x = torch.cat([x, anchor[..., first_missing_phase:]], dim=-1)

        num_periods = x.size(-1) // self.period_len
        periods = x.reshape(*x.shape[:-1], num_periods, self.period_len)
        phase_series = periods.permute(0, 1, 3, 2).contiguous()
        return phase_series, anchor

    @staticmethod
    def center(phase_series, anchor):
        if phase_series.shape[:-1] != anchor.shape:
            raise ValueError("phase_series and anchor shapes are incompatible")
        return phase_series - anchor.unsqueeze(-1)

    @staticmethod
    def restore(displacement, anchor):
        if displacement.shape[:-1] != anchor.shape:
            raise ValueError("displacement and anchor shapes are incompatible")
        return displacement + anchor.unsqueeze(-1)
