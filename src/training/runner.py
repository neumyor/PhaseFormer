"""Shared training protocol for PhaseFormer.

All experiment entry points (official run_*.py, benchmark suite, search runner,
and research scripts) use this module to construct the Lightning Trainer so the
best-checkpoint protocol is defined in exactly one place:

- Lowest validation-loss checkpoint is saved as ``best.ckpt``.
- Callers restore that checkpoint before test evaluation / bad-case export via
  :func:`restore_best_checkpoint`, or pass ``ckpt_path="best"`` to the Trainer
  test method.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def build_logger(save_dir, name="PhaseFormer", version=None):
    """Construct the CSV logger used by all runners."""
    return CSVLogger(save_dir=save_dir, name=name, version=version)


def build_trainer(
    *,
    max_epochs,
    logger,
    patience,
    checkpoint_dir=None,
    ckpt_filename="best",
    accelerator="auto",
    devices=1,
    progress=True,
):
    """Build a Trainer with the canonical best-checkpoint protocol.

    Returns ``(trainer, checkpoint)`` where ``checkpoint`` is the
    ``ModelCheckpoint`` monitoring ``val_loss`` (min). Callers may use
    ``checkpoint.best_model_path`` / ``checkpoint.best_model_score`` after fit.
    """
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=ckpt_filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience),
            checkpoint,
        ],
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=progress,
        log_every_n_steps=1,
        deterministic=True,
    )
    return trainer, checkpoint


def restore_best_checkpoint(model, checkpoint):
    """Restore the lowest-val-loss weights into ``model`` in place.

    Locally generated Lightning checkpoints bundle model config, so restore uses
    ``weights_only=False`` for compatibility with PyTorch's default ``True``.
    """
    if checkpoint is None or not getattr(checkpoint, "best_model_path", None):
        raise RuntimeError("training completed without a best checkpoint")
    state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    return model
