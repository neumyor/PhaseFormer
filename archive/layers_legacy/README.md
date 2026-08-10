# Legacy layer modules (archived)

These files were carried over from the upstream PhaseFormer/reference repository
and are not imported by the active model in `src/models/PhaseFormer.py`, which
only depends on `src.models.layers.SelfAttention_Family`.

They are kept for historical/reference purposes only:

- Autoformer / Crossformer / ETSformer / Pyraformer / Transformer encoder-decoder
  building blocks (`*_EncDec.py`, `AutoCorrelation.py`, `FourierCorrelation.py`,
  `MultiWaveletCorrelation.py`, `Conv_Blocks.py`, `SelfAttention_Family.py`).
- PathFormer modules (`PathFormer_*`) and their helpers.
- `Embed.py`, `revin.py`, `lora.py`, `utils.py` — shared helpers.

If a future model variant needs any of these, restore the file into
`src/models/layers/` (or a `baselines/` subpackage) and add the needed import.
Nothing here is wired into `build_hyperparams` / `PhaseFormer` today.
