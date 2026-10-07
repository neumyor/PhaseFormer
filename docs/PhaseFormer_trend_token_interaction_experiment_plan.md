# PhaseFormer-L low-rank interaction matrix

> **Correction**: the first `trend_token_interaction_20261006` implementation
> used a new Conv1d history encoder and therefore did not test PhaseFormer-L.
> That matrix is invalid for the intended hypothesis. The corrected arms below
> reuse `PooledLowRankWeakPeriodResidualHead` and insert attention only at its
> learned rank-`r` latent.

## Purpose

> **Protocol correction (2026-10-07, user clarification):** the research target
> is architecture comparison with training from scratch, not adaptation of a
> pretrained i0. Audit of the corrected stage-1 remote configs found 210/210
> with an empty `init_checkpoint`, an eight-epoch budget and completed metrics.
> In contrast, all 54 valid stage-2 runs explicitly load trained stage-1 i0
> weights; all 42 stage-3 runs explicitly load stage-2 tp_pre weights. Their
> existing metrics remain valid for those warm-start protocols, but cannot
> establish the best architecture under training from scratch. In particular,
> tp_pre's normalization comparison, rejection of selective/null mechanisms,
> and centered_frozen's advantage over joint adaptation must not be promoted
> to from-scratch architecture decisions. Earlier "best model" wording is
> withdrawn to that extent. Subsequent architecture tests must initialize and
> train all branches jointly from scratch, use matched training/selection
> budgets and seeds, and audit that no trained checkpoint was loaded. Keep
> prior warm-start evidence separate as diagnostics; no new runs are launched
> by this audit.

This round tests whether the learned trend token should only correct the final
forecast, read the phase tokens, write back to the phase tokens, or participate
in repeated/joint token interaction. The existing low-rank PhaseFormer-L
results establish the capacity settings used here; this round changes only the
interaction graph around the legacy residual latent.

## Controlled mechanisms

The arrow denotes information flow. `P` is the routed phase-token sequence and
`T` is one per-channel rank-`r` residual latent, with `r=H/8`.

| Arm | Interaction | Definition |
|---|---|---|
| `phaseformer_l_i0_late_fusion` | none | Legacy rank-`r` latent is decoded independently and fused at the output |
| `phaseformer_l_i1_phase_to_trend` | `P -> T` | Rank-`r` latent queries phase tokens before the unchanged decoder |
| `phaseformer_l_i2_trend_to_phase` | `T -> P` | Phase tokens receive the rank-`r` state before the phase predictor |
| `phaseformer_l_i3_parallel_bidir` | parallel `P <-> T` | Both directions read the original states and update in parallel |
| `phaseformer_l_i6_alternating` | two-round alternating | Two sequential `P -> T -> P` rounds |
| `phaseformer_l_i7_joint` | joint attention | Full self-attention over `[T, P_1, ..., P_M]` |

The phase-to-trend path preserves the original low-rank residual behavior. The
trend-to-phase single-token path is intentionally retained as a broadcast
control: with one key/value token its attention weights are necessarily one.
The joint and alternating arms test whether the phase path benefits from a
state update rather than only from a final residual forecast.

## Frozen protocol

- Data: ETTh1, ETTh2, ETTm1, ETTm2, Weather.
- Lookback: 720; phase period: 24.
- Horizons: 96, 192, 720.
- Seeds: 2021, 2022.
- Loss: Huber; epochs: 8; lowest validation-loss checkpoint.
- Residual head: `PooledLowRankWeakPeriodResidualHead`, `pool_factor=1`,
  `rank=H/8`, `smooth_ratio=0`; the latent is `encoder(pooled(x-x_last))`.
- Readout: unchanged legacy `decoder(r -> H)` plus the `x_last` anchor.
- Fusion gate: static per-channel sigmoid, initialized at 0.2.
- Learning rate: the existing per-dataset/per-horizon base preset; no new grid.
- The primary matrix is validation-only. Test is not read for mechanism or
  hyperparameter selection.

The corrected matrix contains the six requested arms plus a matched `phase_only` control:
`5 datasets × 3 horizons × 2 seeds × 7 arms = 210 jobs`.

## Comparability and reporting

Every arm uses the same pooled-low-rank head, rank, gate initialization, loss,
data split, and seed. Interaction modules are constructed only for the arms that
use them. The report must include parameter counts and wall-clock time because
joint/alternating attention has additional capacity and compute.

The result is an architecture screen. Any later test evaluation must be a
separate, predeclared stage after validation results are frozen; it must not
replace this matrix or be described as an unbiased benchmark.

## Stage 2: Identity Residual and Selective Interaction

User-authorized on 2026-10-07. Implement and test three hypotheses: preserve the
rank latent with a pre-norm residual; let the trend reject phase messages with a
zero-valued null; let each phase choose between the trend and a zero message.
One rank-r trend token, the original low-rank encoder/decoder, phase backbone,
and static output fusion are retained. No Conv1d trend encoder is introduced.

Pilot settings: ETTh1-H96, Weather-H720, ETTm2-H720, each seed 2021/2022.
All arms start from the same completed stage-1 i0 checkpoint within a setting.
All trainable parameters are updated for four additional epochs with a fresh
Adam optimizer and the source learning rate/scheduler, batch size and Huber
loss. The source config and checkpoint hash are saved. Lowest validation-loss
checkpoint after training is used for every arm (no candidate-only initial
checkpoint fallback). This is matched fine-tuning, not a from-scratch benchmark.

| Arm | Update | Purpose |
|---|---|---|
| i0_continue | no interaction | Control for extra training |
| pt_post / tp_post | legacy LN(z + message) | Matched legacy update controls |
| pt_pre / tp_pre | z + F(LN(z), LN(source)) | Normalization-position controls |
| pt_scaled / tp_scaled | z + alpha F(LN(z), LN(source)) | Proposal 1, alpha starts at zero |
| pt_null | pt_scaled with zero-valued null competitor | Proposal 2 |
| tp_selective | tp_scaled with per-phase trend/null choice | Proposal 3 |

This gives 54 runs. First complete the 42 baseline/update-control jobs, then
the 12 routing jobs; do not combine null read and selective write in this pilot.
Pre-norm arms share legacy attention projections. Null mass scales all output
biases too, making a fully rejected message exactly zero. Scaled arms preserve
the full i0 output exactly at initialization. The source weight loader rejects
unexpected keys or missing common weights, and a real validation batch checks
initial identity before each scaled-arm run. The seed is reset immediately
before fit to align data order independently of constructor RNG consumption.

Second verification: source diff review, configuration/full-output equivalence,
null rejection with nonzero biases, selective-query gradients, residual-scale
gradient startup, broadcast control, and explicit attention vs MHA equivalence.
Then run all nine arms through a two-train-batch/two-validation-batch integration
smoke before launching the complete pilot on the eight available A800 GPUs.

Canonical output: `research_runs/lowrank_interaction_stage2_20261007_v2/`, containing
the six audit files and referenced figures only. Checkpoints/logs live in the
separate ignored `research_runs/lowrank_interaction_stage2_20261007_v2_scratch/`.
Entrypoint: `scripts/run_lowrank_interaction_stage2.py --source-dir
research_runs/phaseformer_l_interaction_20261006 --scratch-dir
research_runs/lowrank_interaction_stage2_20261007_v2_scratch --audit-dir
research_runs/lowrank_interaction_stage2_20261007_v2 --gpus 0,1,2,3,4,5,6,7`.

The first stage-2 batch without the v2 suffix is INVALID: the initialization
audit left the model in eval mode before fit (22 result files at interruption).
Code `0f14e9e` explicitly calls train() after the audit and adds a per-training-
batch assertion that all modules are in training mode. All nine integration
smokes were repeated successfully before the v2 launch. Earlier smoke results
establish pipeline behavior only and cannot establish training efficacy.

Post-training paired analysis ranks sample/channel MAE, selects top five high
baseline error, regression and improvement cases with a within-channel start
distance of at least H per category, and saves forecasts only for selected cases.
Every fixed candidate is retained and compared to i0_continue; pt_null is also
compared to pt_scaled and tp_selective to tp_scaled. Capture null acceptance,
within-sample phase-gate spread, raw/effective message norms, and same-checkpoint
message-disabled predictions. Gate/entropy correlations are descriptive, not
causal claims. Setting coverage, error means, ranking, NPZ predictions, figure
references and ZIP bytes must be verified before reporting completion.
No test loader or hyperparameter search is permitted in this pilot. No automatic
full-matrix expansion is planned before assessing pilot evidence.

### Stage 2 Completed Results (2026-10-07)

The valid v2 matrix completed all 54 runs with zero failures on code `0f14e9e`.
The paired audit passed: 3,050,768 sample/channel comparison rows, 720 selected
cases and 24 figures. All six canonical files were downloaded and their SHA256
hashes match the remote originals. The local directory whitelist, all 54 result
rows, setting coverage and report/ZIP byte consistency were independently checked.
The earlier unsuffixed eval-mode batch remains invalid and is excluded below.

Setting-equal candidate-minus-`i0_continue` validation differences (negative is
better; wins count any negative MSE difference, including very small differences):

| Arm | Mean delta MAE | Mean delta MSE | MSE wins |
|---|---:|---:|---:|
| pt_post | +0.009651 | +0.016683 | 2/6 |
| pt_pre | +0.000289 | +0.000410 | 2/6 |
| pt_scaled | -0.000140 | +0.000167 | 3/6 |
| tp_post | +0.000551 | +0.001539 | 2/6 |
| tp_pre | -0.000270 | -0.000606 | 4/6 |
| tp_scaled | -0.000225 | -0.000379 | 5/6 |
| pt_null | +0.000174 | +0.000515 | 2/6 |
| tp_selective | +0.000458 | +0.000869 | 1/6 |

**Proposal decisions**:

- Retain pre-norm T->P as the strongest direction for further investigation.
  `tp_pre - tp_post` has mean delta MSE -0.002145 and improves all six settings.
  However, `tp_scaled - tp_pre` is +0.000227, with only 2/6 improvements: the
  evidence does not establish an additional average benefit from zero-initialized
  residual scaling. In P->T, pre-norm largely removes the ETTh1 warm-start penalty
  (seed 2021 MSE: i0 0.672340, post 0.721411, pre 0.672266), but does not provide
  a consistent gain across settings. This is evidence about the update design in
  matched fine-tuning, not a general claim that post-norm is always harmful.
- Do not expand null read in its current form. `pt_null - pt_scaled` is
  +0.000314 MAE / +0.000348 MSE; only 2/6 MSE improvements, both tiny ETTm2
  differences. The rejection option has no demonstrated incremental benefit.
- Do not expand selective write in its current form. `tp_selective - tp_scaled`
  is +0.000683 MAE / +0.001249 MSE, with 0/6 improvements. Phase-dependent
  receipt alone did not improve upon the corresponding broadcast control.

**Activation and sample evidence**:

- P->T null real-phase acceptance is 0.951--0.963 on ETT, close to the 24/25
  acceptance of equal phase/null scores. Weather acceptance is 0.459/0.670,
  showing more rejection, but both seeds are worse than `pt_scaled` in MSE.
  Rejection magnitude is not established as a measure of reliability.
- Selective-write phase-gate standard deviation is 0.071/0.120 on ETTm2, but
  only 0.0071/0.0077 on Weather. Thus ETTm2 shows phase differentiation while
  Weather remains nearly a common receipt magnitude. ETTm2 effective update
  norms are 0.466/0.554, and same-checkpoint enabled-minus-disabled MSE is
  +0.001131/+0.000479. Weather's corresponding MSE is -0.006392/-0.005379,
  yet it still loses to the trained broadcast control. Reliance on a message
  within one checkpoint does not establish superiority over another model.
- The programmatically selected Weather-H720 seed2021 null-read regression
  `1910,c0` has acceptance approximately 1 and delta MSE +2.35810 versus i0;
  disabling its message at fixed weights reduces MSE by 3.0895. The improvement
  `2111,c14` has acceptance 0.025 and delta MSE -1.28771, but the enabled-minus-
  disabled difference is only -0.00037. Its improvement versus i0 cannot be
  attributed to the current message alone.
- Weather-H720 seed2021 selective-write regression `2577,c15` has acceptance
  approximately 1, delta MSE +4.04697 versus i0 and enabled-minus-disabled MSE
  +3.19742. ETTm2-H720 seed2022 regression `7538,c4` instead has acceptance
  0.063 and delta MSE +0.25369, while enabling the message improves MSE by
  0.04355. These cases rule out a simple reading of low acceptance as successful
  protection or of every regression as direct harm from the current message.

These observations motivate keeping the identity bypass and studying message
content before adding more routing freedom. They do not identify a proven causal
failure mechanism. The pilot used four extra training epochs from selected i0
checkpoints and reused validation for checkpoint selection and analysis. It is
not a from-scratch comparison, a full ETT/Weather benchmark, a significance test,
or evidence of improvement over Golden/test results. No automatic expansion is
launched; i0 remains the established low-rank baseline.

Canonical report and portable image bundle:
`research_runs/lowrank_interaction_stage2_20261007_v2/objective_error_analysis.md`
and `objective_error_analysis.zip` in the same directory. Source checkpoints,
invalid-run evidence and v2 training records remain outside the canonical audit
directory for reproducibility; they are not committed to Git.

### Frozen-checkpoint mechanism audit (2026-10-07)

The user requests finer sample/activation analysis and reassessment of the
research hypotheses. No training or test evaluation is added. Use the valid v2
checkpoints only. `scripts/audit_lowrank_stage2_mechanisms.py` evaluates all eight
interaction arms on 128 equally spaced validation windows per setting, with all
channels. This cohort is selected independently of errors and is not a new full
validation efficacy estimate. Adjacent windows may overlap.

Ten detailed cases are selected from the existing audit's rank-one MAE cases:
ETTh1-H96 both seeds, pt_post regression; Weather-H720 seed2021, pt_null and
tp_selective regression/improvement; ETTm2-H720 seed2022, tp_selective
regression/improvement; Weather-H720 seed2022, tp_scaled regression/improvement.
The same windows/channels are also evaluated across all eight arms to prevent
confusing differences between samples with differences between mechanisms.

Frozen interventions: message bypass; 0.25/0.5/2 times message scale; removal of
null with real-key renormalization; uniform phase read with original acceptance;
constant per-sample mean phase gate; a 12-slot cyclic shift of gates preserving
their histogram; and separate post-norm message-only versus norm-only paths.
Native and bypass implementations must match the original model at every batch.
ETTh1 source-i0 insertion checks additionally separate the initial normalization
shock from later model adaptation. Use highest float32 precision, documenting
the difference from the original medium-precision efficacy evaluation.

Measure both latent changes and output corrections. For d=y_on-y_off and
e=y_true-y_off, check the exact identity delta MSE=mean(d^2)-2 mean(d*e).
The label-dependent optimum along d is a diagnostic upper bound, never an
inference-time gate or an authorized model selection rule. Full-validation
gate/error correlations are reused from the previous audit and reported as
observational. Hold weights fixed, and qualify all activation-distribution shift
and overlapping-window limitations.

Output: `research_runs/lowrank_stage2_mechanism_audit_20261007/`, in the canonical
six-file format with ten figures. Analysis intermediates use a separate ignored
directory. Existing source weights and evidence are not changed.

#### Mechanism audit outcome

Completed 72,912 sample/channel/arm records and ten six-panel figures. All six
settings have native-output max_abs=0 against the original forward. Off-path,
gate-mean preservation and linear P->T half-scale identities passed. The final
report adds a Chinese hypothesis review and next-step rationale in sections 9/10;
this editorial interpretation is separate from the script-generated numeric
sections. No training or test evaluation was performed.

- **Identity bypass is supported in this warm-start context.** At source-i0
  insertion on ETTh1 seed2021 `485,c5`, off/message-only/norm-only/post-norm MSE
  is 1.357256/1.358235/4.341222/4.339780. Seed2022 `533,c5` reproduces the same
  pattern (1.221824/1.235505/4.502841/4.503758). This directly identifies LN as
  the dominant insertion perturbation for those samples, not as a universal
  explanation of every trained-model difference. The additional efficacy of
  zero-initialized scale remains unproven against the pre-norm-only control.
- **Attention receipt is not a calibrated reliability measure.** Weather
  seed2021 null-read `1910,c0` accepts nearly all mass and has output correction
  cosine -0.959, native/off MSE 4.340156/1.250325. Conversely ETTm2 seed2022
  selective-write `7538,c4` accepts only 0.0632, but native/off/no-null MSE is
  0.587408/0.630951/0.143215. Rejection can suppress a useful correction.
  Removing rejection is not universally beneficial: its diagnostic-cohort
  effect differs across settings, so these examples do not justify removing
  all gates or claiming that no rejection mechanism could work.
- **Useful phase alignment is not demonstrated.** Constant mean receipt has
  lower diagnostic-cohort MSE in all six settings; a 12-slot gate shift has
  lower MSE in five. Many differences are tiny. The current update matrix over
  phase slots has rank at most one, delta_p_j=alpha*g_j*v(t): its selectivity
  varies magnitude, not direction. It should not be described as phase-specific
  message content.
- **Magnitude control alone is insufficient.** Wrong-direction examples need
  different content or rejection, while `7538,c4` and `8142,c3` improve when
  more of their current correction is allowed. Large token deltas and small
  deltas can both help or hurt. Output-space alignment and energy are more
  informative than raw latent norms, but require future labels for diagnosis.
- **Training adaptation must be distinguished from current message efficacy.**
  Weather selective-write `1955,c7` has i0/native/off MSE
  1.012050/0.215311/0.214392. The large cross-model improvement is not evidence
  that the current message helps. An off checkpoint also differs from a model
  trained without interaction, so this is not a full causal decomposition.

Next steps, not yet implemented: first compare frozen-i0 interaction-only
learning with matched joint fine-tuning. Then test phase-conditioned message
content using a small bilinear/Hadamard interaction from the same single rank-r
trend token, with a centered-phase variant and a matched uncentered control.
If reliability gating is revisited, use historical hindcast errors/disagreement
to assess expected correction benefit, and validate calibration on held-out
time intervals. Consider output-correction energy rather than latent norm for
amplitude limits; retain separate direction/scale ablations. No oracle scale
derived from the future labels may enter inference. These are new hypotheses,
not conclusions that the proposed changes will improve forecasting.

## Stage 3: Phase-conditioned content on tp_pre (2026-10-07)

User authorizes implementation and tests of the preceding content hypothesis.
Use the valid stage-2 `tp_pre` checkpoint for each of the six existing settings:
ETTh1-H96, ETTm2-H720, Weather-H720, seeds 2021/2022. Keep original pooled-lowrank
E/D, one rank-r trend token, r=H/8, phase backbone, pre-norm broadcast and static
output fusion. No Conv1d, additional trend tokens, null gate, output norm penalty
or additional rejection mechanism is introduced in this stage.

Add u_j=Wo[Wt(LN(t))*Wp(LN(p_j))], width 8, and update p_j'=p_j+m0+beta*u_j.
The centered candidate uses u_j-mean_k(u_k). beta starts at zero so the complete
initial output exactly equals source tp_pre. Both conditional variants have
identical parameters. The trend-only capacity control replaces LN(p_j) with
adaptive pooling of LN(t) to the phase width, broadcast to every phase; it uses
the same projection shapes and parameter count without unused parameters.
Context and output projections omit biases that would cancel under centering.
This control matches capacity but does not uniquely match every optimization
property; trend-only performance must be reported, not assumed inferior.

Seven trained arms per setting: tp_pre_continue; trend_frozen, phase_frozen,
centered_frozen; trend_joint, phase_joint, centered_joint. Frozen arms train only
the new content module while all original weights/buffers remain fixed and all
original modules stay in eval mode (including dropout). Joint arms train all
parameters in normal train mode. All start from the same tp_pre checkpoint,
receive four epochs with fresh Adam, source LR/scheduler, batch size, loss and
data order seed, and select minimum validation Huber after training. Source
tp_pre is also evaluated as the frozen baseline. Frozen and joint comparisons
are reported separately; a joint gain is not attributed to the new message
unless fixed-weight message ablations support that narrower claim.

Total: 42 training jobs plus six unchanged source evaluations. Complete baseline
and frozen diagnostics before joint candidates. Nine existing stage-2 mechanism
tests and new identity/gradient/centering/capacity/alignment checks precede seven
arm training smokes. Each job checks source SHA256, exact common-weight loading,
full-model initial identity, training mode, and (frozen arms) final bitwise
equality of every source state tensor. No automatic hyperparameter sweep.

Hypotheses and failure conditions:

1. Phase-conditioned content should improve over matched tp_pre continuation
   and over equal-parameter trend-only expansion, across seeds/settings. Without
   both comparisons, added capacity cannot be excluded.
2. Centering should add value over the uncentered conditional arm. Latent zero
   mean is checked explicitly but is not assumed to preserve output mean.
3. Frozen-arm gains establish that the new message can help without modifying
   old weights; joint-only gains are labeled adaptation-dependent.
4. At fixed trained weights, disabling just the new content, or cyclically
   shifting/reversing its phase context, should worsen predictions if the new
   content and its alignment matter. The trend-only control must be invariant
   to phase permutations. Keep the old tp_pre broadcast enabled in every such
   diagnostic. Off-path frozen predictions must equal source tp_pre.
5. Revisit Weather-H720 seed2021 1910,c0 and 2577,c15 as prespecified anchor cases:
   reduced harm on the former must not silently erase benefit on the latter.
   Also inspect 1955,c7 and ETTm2-H720 seed2022 7538,c4 / 8142,c3. These targeted
   checks do not replace programmatic top-five MAE regression/improvement/high
   baseline error selection per setting/arm.

Full validation inference records paired MAE/MSE, horizon thirds, content raw
and effective update norms, phase mean/spread, output-correction energy and
alignment, and enabled/off/phase-permuted predictions. The same validation set
has already influenced source model selection and hypotheses; this is an
exploratory matched continuation, not a blind or from-scratch benchmark. No test
or Golden superiority claim, statistical significance claim, or output-space
constraint experiment is authorized by this fixed matrix. Output constraints
remain a separately testable follow-up if needed.

Entrypoint: `scripts/run_lowrank_stage3.py --source-scratch
research_runs/lowrank_interaction_stage2_20261007_v2_scratch --scratch-dir
research_runs/lowrank_stage3_20261007_scratch --audit-dir
research_runs/lowrank_stage3_20261007 --gpus 0,1,2,3,4,5,6,7`.
Canonical output has the six audit files and referenced figures only. Training
records/checkpoints are kept separately in the ignored scratch directory.

### Stage 3 completed: hypothesis decisions

All 42 training jobs completed with zero formal failures on `a8b2026`.
Full-validation audit passed: 2,669,422 paired rows, 665 selected case records
(including 35 prespecified anchor records), 24 figures. Source and continued
tp_pre references yield 48 aggregate rows. Thirteen unit checks, all seven
training smokes, initial full-output identity, frozen-state equality and final
off=source/permutation/centering invariants passed. `93fae7c` adds symmetric
activation heatmap colors and selected-case update direction diversity; it does
not change training or inference results.

Setting-equal deltas, negative better. Frozen arms compare against source tp_pre;
joint arms compare against matched tp_pre_continue:

| Arm | Mean delta MAE | Mean delta MSE | MSE wins |
|---|---:|---:|---:|
| trend_frozen | +0.000777 | +0.000536 | 3/6 |
| phase_frozen | +0.000670 | +0.000688 | 1/6 |
| centered_frozen | -0.000222 | -0.000290 | 5/6 |
| trend_joint | +0.000866 | +0.000533 | 2/6 |
| phase_joint | +0.000311 | +0.000392 | 2/6 |
| centered_joint | -0.000034 | +0.000205 | 2/6 |

The continued baseline itself has mean delta MSE +0.001139 versus source tp_pre
(2/6 wins). centered_joint is +0.001344 versus source. Thus neither additional
training nor unfreezing is assumed beneficial. Results are a four-epoch pilot,
not a claim that joint training can never work.

1. **Centering: supported in this protocol.** centered_frozen improves over
   phase_frozen by -0.000978 mean MSE, 6/6 settings; centered_joint improves over
   phase_joint by -0.000187, 4/6. Uncentered phase content often mainly changes
   the common phase component: Weather frozen mean-token norm is 0.206546 vs
   total per-token norm 0.206647 in seed2021, and 0.117665 vs 0.117706 in seed2022.
2. **Phase content beats capacity: partially supported, not universal.**
   centered_frozen minus trend_frozen is -0.000826 on average, but only 3/6 wins.
   Uncentered phase_frozen is worse than trend_frozen on average (+0.000152).
   Added phase dependence alone is insufficient.
3. **Independent useful correction: supported in 5/6 settings.** Frozen off
   exactly equals source tp_pre throughout validation, and old state tensors
   are unchanged. Its improvements therefore occur without old-weight
   adaptation. ETTh1 seed2022 has a small regression (+0.000013 MSE).
4. **Phase alignment: supported for the useful centered corrections.** Both
   rolling and reversing phase context worsen all five improving frozen
   settings. ETTm2 seed2022: native-minus-off -0.001188, roll-minus-native
   +0.002931, reverse-minus-native +0.002702. The non-improving ETTh1 seed2022
   also fails this alignment check. In joint models, native beats off and both
   permutations in all six settings, yet whole-model MSE fails to beat matched
   continuation on average. Useful message and useful joint training differ.
5. **Targeted error repair: partial and small.** For Weather seed2021, frozen
   centered source->candidate MSE is 2.095243->2.095060 at 1910,c0,
   3.336931->3.322840 at 2577,c15 and 0.236906->0.224013 at 1955,c7.
   The first anchor's main level error is not solved. ETTm2 seed2022 7538,c4
   improves 0.365061->0.363035, whereas 8142,c3 worsens 0.744187->0.747560.

**Mechanistic refinement:** all six source predictors are shared linear maps,
with no phase alignment module; RevIN and fusion are channelwise affine. For
these exact settings, centered latent updates imply zero summed correction over
the 24 phases of each future cycle in real arithmetic. This is more specific
than the earlier general warning for nonlinear predictors. A highest-precision
check on eight validation windows per setting gives maximum absolute cycle-mean
change 3.97e-8--7.45e-8. The medium-precision efficacy runs can show small rounding
departures. The learned addition is therefore primarily a cycle-shape correction
preserving the old cycle mean, not a solution for common level/trend errors.

Available multi-direction capacity is not yet proved necessary. Selected-case
participation rank (sum sigma^2)^2/sum sigma^4 averages approximately 1.00--1.39
across settings; many updates remain dominated by one direction. A future
centered signed-scalar control and a centered phase-only control are needed to
separate the benefits of centering, direction diversity and trend conditioning.
These controls are recommendations, not new experiments launched in this stage.

**Decision:** retain centered_frozen as a small, attributable refinement of the
trained tp_pre checkpoint, while retaining tp_pre as its baseline. It adds 169
parameters on ETTh1 and 857 on ETTm2/Weather. Do not adopt joint unfreezing or claim
a general benchmark win. Canonical evidence is
`research_runs/lowrank_stage3_20261007/`; selection reused validation, no test
was read, and the original PhaseFormer-L rank-r encoder/decoder remains intact.
