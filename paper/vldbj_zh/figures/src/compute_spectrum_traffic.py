"""Spectrum of the Traffic dataset: amplitude vs. phase (period) length.

Run on the server (CPU only, numpy/pandas):
    python compute_spectrum_traffic.py <traffic.csv> <out.csv>

Method: rFFT of every channel over the full series (length L = 17544, no
train/val/test split, no normalisation, DC bin dropped), amplitude |rfft|
averaged over the 862 channels. Frequency index k maps to phase length
p = L / k; for each integer p in [2, 50] we read the bin k = round(L / p).
The top-5 phase lengths by amplitude are marked in the CSV (rank column).
"""
import sys
import numpy as np
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
X = pd.read_csv(src).iloc[:, 1:].values.astype(np.float64)  # drop date column
L, C = X.shape
amp = np.abs(np.fft.rfft(X, axis=0)).mean(axis=1)  # mean over channels

rows = []
for p in range(2, 51):
    k = int(round(L / p))
    rows.append((p, k, amp[k]))
df = pd.DataFrame(rows, columns=["phase_length", "rfft_bin", "amplitude"])
df["rank"] = df["amplitude"].rank(ascending=False, method="first").astype(int)
df.loc[df["rank"] > 5, "rank"] = 0  # 0 = not a top-5 peak
df.to_csv(dst, index=False, float_format="%.6f")
print("L=%d channels=%d" % (L, C))
print(df[df["rank"] > 0].sort_values("rank").to_string(index=False))
