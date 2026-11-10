"""Official efficient-forecasting baselines, vendored for the component-MSE run.

The server has no network access, so the model code is copied here from the
authors' repositories and only trimmed of unrelated imports:

* DLinear    -- thuml/Time-Series-Library, ``models/DLinear.py`` (Zeng et al. 2023)
* SparseTSF  -- lss-1138/SparseTSF, ``models/SparseTSF.py`` (Lin et al. 2024)
* FITS       -- VEWOXIC/FITS, ``models/FITS.py`` (Xu et al. 2024)
* CycleNet   -- ACAT-SCUT/CycleNet, ``models/CycleNet.py`` (Lin et al. 2024)

Architecture hyperparameters follow each paper's per-dataset scripts (Weather:
SparseTSF period 4, CycleNet cycle 144, FITS cut_freq 72; Traffic: SparseTSF
period 24, CycleNet cycle 168, FITS cut_freq 72). Training is the shared
protocol of ``train_baseline.py`` (Adam 1e-3, Huber, 30 epochs, patience 8),
which is *not* each paper's recipe, so these numbers compare models to each
other and to PhaseFormer rather than reproducing published tables.

Every model maps ``(B, L, C)`` to ``(B, H, C)`` on the standardized data.
CycleNet additionally needs the phase index of each forecast origin; see
``needs_cycle_index`` and ``cycle_phase``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WEATHER, TRAFFIC = 'Weather', 'Traffic'
ETT_HOUR = ('ETTh1', 'ETTh2')
ETT_MINUTE = ('ETTm1', 'ETTm2')
ETT = ETT_HOUR + ETT_MINUTE

# Per-dataset architecture settings taken from the official run scripts.
CYCLE_LENGTH = {WEATHER: 144, TRAFFIC: 168,
                'ETTh1': 24, 'ETTh2': 24, 'ETTm1': 96, 'ETTm2': 96}
SPARSE_PERIOD = {WEATHER: 4, TRAFFIC: 24,
                 'ETTh1': 24, 'ETTh2': 24, 'ETTm1': 4, 'ETTm2': 4}
CUTOFF_FREQ = 72          # int(720 // 24 + 1) * 2 + 10, the authors' default for lookback 720
MOVING_AVG = 25
CYCLE_MODEL_TYPE = 'linear'
SPARSE_MODEL_TYPE = 'linear'


class _Config:
    """The attribute bag the official ``Model`` constructors read."""

    def __init__(self, **values):
        self.__dict__.update(values)


# --------------------------------------------------------------------------
# DLinear (TSLib)
# --------------------------------------------------------------------------
class _MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = _MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = _SeriesDecomp(MOVING_AVG)
        self.individual = individual
        self.channels = channels
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[-1].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[-1].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        return (seasonal_output + trend_output).permute(0, 2, 1)


# --------------------------------------------------------------------------
# SparseTSF (official)
# --------------------------------------------------------------------------
class SparseTSF(nn.Module):
    def __init__(self, seq_len, pred_len, channels, period_len, model_type=SPARSE_MODEL_TYPE):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = channels
        self.period_len = period_len
        self.model_type = model_type
        assert self.model_type in ['linear', 'mlp']
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1,
                                kernel_size=1 + 2 * (self.period_len // 2), stride=1,
                                padding=self.period_len // 2, padding_mode='zeros', bias=False)
        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.linear(x)
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        return y.permute(0, 2, 1) + seq_mean


# --------------------------------------------------------------------------
# FITS (official)
# --------------------------------------------------------------------------
class FITS(nn.Module):
    def __init__(self, seq_len, pred_len, channels, cut_freq=CUTOFF_FREQ, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = channels
        self.dominance_freq = cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for _ in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)
        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0
        low_specx = low_specx[:, 0:self.dominance_freq, :]
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
        low_specxy = torch.zeros(
            [low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio
        return (low_xy * torch.sqrt(x_var) + x_mean)[:, -self.pred_len:]


# --------------------------------------------------------------------------
# CycleNet (official)
# --------------------------------------------------------------------------
class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class CycleNet(nn.Module):
    def __init__(self, seq_len, pred_len, channels, cycle, model_type=CYCLE_MODEL_TYPE, use_revin=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = channels
        self.cycle_len = cycle
        self.model_type = model_type
        self.use_revin = use_revin
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)
        x = x - self.cycleQueue(cycle_index, self.seq_len)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean
        return y


# FITS is configured per dataset by its authors' scripts: channel-independent
# parameters on Weather (``--individual``) and shared parameters on Traffic,
# with the cutoff swept over harmonic orders 5/8/10/12 for lookback 720.
FITS_INDIVIDUAL = {WEATHER: True, TRAFFIC: False,
                   'ETTh1': False, 'ETTh2': False, 'ETTm1': False, 'ETTm2': False}
# H_order 8 at lookback 720 for Weather/Traffic, validated on val; the ETT values
# are the middle of the authors' sweeps (2-6 for hourly, 6-14 for 15-minute) and
# are re-validated by the sweep recorded in docs/component_mse_ETT.md.
FITS_CUTOFF = {WEATHER: 258, TRAFFIC: 258,
               'ETTh1': 134, 'ETTh2': 134, 'ETTm1': 258, 'ETTm2': 258}


def fits_cutoff(seq_len, order):
    return int(seq_len // 24 + 1) * order + 10


BUILDERS = {
    'DLinear': lambda seq_len, pred_len, channels, dataset: DLinear(seq_len, pred_len, channels),
    'SparseTSF': lambda seq_len, pred_len, channels, dataset: SparseTSF(
        seq_len, pred_len, channels, SPARSE_PERIOD[dataset]),
    'FITS': lambda seq_len, pred_len, channels, dataset: FITS(
        seq_len, pred_len, channels, cut_freq=FITS_CUTOFF[dataset],
        individual=FITS_INDIVIDUAL[dataset]),
    'CycleNet': lambda seq_len, pred_len, channels, dataset: CycleNet(
        seq_len, pred_len, channels, CYCLE_LENGTH[dataset]),
}
BASELINES = tuple(BUILDERS)
NEEDS_CYCLE_INDEX = {'CycleNet'}


def build(model, seq_len, pred_len, channels, dataset):
    if model not in BUILDERS:
        raise ValueError(model)
    return BUILDERS[model](seq_len, pred_len, channels, dataset)


def cycle_phase(timestamps, dataset):
    """Phase of each forecast origin inside the model's cycle, from real timestamps.

    Weather is 10-minute data with a 144-step daily cycle; Traffic is hourly with
    the 168-step weekly cycle the authors use. Anchoring to the calendar keeps
    train, validation and test phases consistent across the split boundaries.
    """
    stamps = np.asarray(timestamps).astype('datetime64[m]')
    days = stamps.astype('datetime64[D]')
    minutes = (stamps - days).astype('int64')
    if dataset == WEATHER:
        return (minutes // 10).astype('int64') % CYCLE_LENGTH[WEATHER]
    if dataset in ETT_MINUTE:
        return (minutes // 15).astype('int64') % CYCLE_LENGTH[dataset]
    if dataset in ETT_HOUR:
        return (minutes // 60).astype('int64') % CYCLE_LENGTH[dataset]
    weekday = (days.astype('int64') + 3) % 7                      # Monday = 0
    hours = minutes // 60
    return (weekday * 24 + hours).astype('int64') % CYCLE_LENGTH[TRAFFIC]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
