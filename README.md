# PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting

This repository implements the core methods of the paper **“PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting”**, built with PyTorch Lightning for training and evaluation.

Compared with traditional patch/segment-based paradigms, PhaseFormer models in the **phase domain**:

- Splits time series into fixed cycles `period_len` and encodes them into *phase tokens*.
- Interacts in the phase space through a **cross-phase routing** mechanism.
- Stacks **phase blocks** layer by layer to predict future phases and reconstruct them back into time series.

---

## 🛠 Installation & Environment

- Python ≥ 3.9  
- PyTorch ≥ 2.1 (GPU acceleration recommended)  
- PyTorch Lightning ≥ 2.1  
- Other dependencies: `pandas`, `numpy`, `scikit-learn`, `easydict`

The repository uses `uv` to lock a reproducible environment:

```bash
uv sync
```

Run commands through the locked environment:

```bash
uv run python run_etth1.py --horizons 96 --mode latest
```

The optional GIFT-Eval/GluonTS integration is installed separately:

```bash
uv sync --group gift-eval
```

---

## 📂 Data Preparation

Place datasets under `resources/all_datasets/` in the following subdirectories (empty folders are already included for guidance):

- `ETT-small/`: `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
- `electricity/`: `electricity.csv`
- `traffic/`: `traffic.csv`
- `weather/`: `weather.csv`

Dataset paths are centrally managed in `src/dataset/data_info.py`. For example:

```python
"ETTh1": {
    "data": "ett_h",
    "data_path": "ETTh1.csv",
    "batch_size": 256,
    "root_path": "./resources/all_datasets/ETT-small",
    "num_variants": 7,
}
```

You can override the default paths or batch sizes directly in the run scripts if needed.

---

## 🚀 Quick Start

This repository provides ready-to-run scripts with recommended hyperparameters for each dataset.
Make sure the corresponding data is available before running.

- **Run ETTh1:**

```bash
uv run python run_etth1.py
```

This will train and evaluate on `pred_len ∈ {96, 192, 336, 720}` and export logs & test results to `./log/training_results/PhaseFormer/`.

- **Run Electricity:**

```bash
uv run python run_electricity.py
```

It will run on the four forecast horizons and export a result summary CSV.

All official dataset entry points, including Traffic, share the same CLI. Use
`--mode original` for the original phase-only policy and `--mode latest` for
the evidence-backed preset. `--horizons` accepts `all` or a comma-separated
list such as `96,192`.

Training summaries report the effective loss name. Legacy configurations with
`use_huber_loss=true` are normalized to `loss_func=huber`, so a run cannot be
labelled as MSE while optimizing Huber loss.

Early stopping and final evaluation use the checkpoint with the lowest
validation loss. Results produced before this protocol was introduced used the
last training epoch and are not directly comparable without a matched rerun.

---

## 📊 Results & Logs

- Training progress and per-epoch evaluation metrics (MAE/MSE) are logged by Lightning’s `CSVLogger`:
  - Location: `./log/training_results/PhaseFormer/<dataset-lookback-horizon-...>/`
- After completion, summary CSV files are generated:
  - ETTh1: `summary_etth1_*.csv`
  - Electricity: `summary_electricity_*.csv`

---

## 🔑 Key Information for Reproducibility

- **Device:** Default `devices=[0]` (single GPU). Modify the Trainer config for multi-GPU or CPU.
- **Default window:** `seq_len=720`, `period_len=24` (customizable in run scripts).
- **Metrics:** MSE / MAE / RMSE / MAPE (see `src/utils/metrics.py`).
