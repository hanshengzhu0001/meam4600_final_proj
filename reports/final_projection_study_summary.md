# Final Project Study Summary (Tracked Report)

Date: 2026-04-22  
Repo: `meam4600_final_proj`  
Branch: `study/projection-sweep`  
Base commit used for study: `a0b111e`  
Current HEAD before this update: `50cfed8`

## 1. Best Checkpoint(s)

- Best overall inverse-recovery checkpoint: `outputs/posterior_projection_e200_big/best.pt`
- Strong baseline reference: `outputs/posterior_projection_baseline/best.pt`
- Additional evaluated checkpoint (did not win): `outputs/posterior_projection_e400_big_b128/best.pt`

## 2. Evaluation Tables (JSON + CSV)

Full `128 x 3` ranking tables:
- `outputs/posterior_projection_baseline/eval_full.json`
- `outputs/posterior_projection_baseline/eval_full.csv`
- `outputs/guidance_big_g2.0/eval_full.json`
- `outputs/guidance_big_g2.0/eval_full.csv`
- `outputs/guidance_big_g4.0/eval_full.json`
- `outputs/guidance_big_g4.0/eval_full.csv`
- `outputs/guidance_big_g8.0/eval_full.json`
- `outputs/guidance_big_g8.0/eval_full.csv`
- `outputs/guidance_big_g10.0/eval_full.json`
- `outputs/guidance_big_g10.0/eval_full.csv`
- `outputs/guidance_big_g14.0/eval_full.json`
- `outputs/guidance_big_g14.0/eval_full.csv`
- `outputs/guidance_big_g16.0/eval_full.json`
- `outputs/guidance_big_g16.0/eval_full.csv`
- `outputs/guidance_e400bigb128_g4.0/eval_full.json`
- `outputs/guidance_e400bigb128_g4.0/eval_full.csv`

Medium `32 x 2` step-count checks:
- `outputs/guidance_big_g8.0_steps200/eval_32x2.json`
- `outputs/guidance_big_g8.0_steps200/eval_32x2.csv`
- `outputs/guidance_big_g4.0_steps200/eval_32x2.json`
- `outputs/guidance_big_g4.0_steps200/eval_32x2.csv`

Medium `32 x 2` guidance sweep extensions:
- `outputs/guidance_big_g6.0/eval_32x2.json`
- `outputs/guidance_big_g6.0/eval_32x2.csv`
- `outputs/guidance_big_g10.0/eval_32x2.json`
- `outputs/guidance_big_g10.0/eval_32x2.csv`
- `outputs/guidance_big_g12.0/eval_32x2.json`
- `outputs/guidance_big_g12.0/eval_32x2.csv`
- `outputs/guidance_big_g14.0/eval_32x2.json`
- `outputs/guidance_big_g14.0/eval_32x2.csv`
- `outputs/guidance_big_g16.0/eval_32x2.json`
- `outputs/guidance_big_g16.0/eval_32x2.csv`

Note: `outputs/` is gitignored, so these artifacts are local cluster outputs by design.

### Metric coverage update (2026-04-22)

The evaluation pipeline now reports distribution-level metrics for each schedule/order row:
- `mmse`: mean-squared error between generated vs reference sample means (state-space, joint `[u; v]`)
- `smse`: mean-squared error between generated vs reference sample standard deviations
- `fpd`: Fr\'echet-style distance between generated vs reference state distributions

Constraint-error fields are now explicitly separated:
- `ce_ic`: initial-condition constraint error
- `ce_bc`: boundary-condition constraint error
- `ce_cl`: conservation-law / governing-equation constraint error

For non-applicable constraints (for example `ce_ic` on this static elliptic benchmark), the pipeline emits `NaN`; report tables should render these as `N/A`.

## 3. Objective Winners (Full Study)

Compared full runs:
- baseline (`80ep`)
- `e200_big + g=2.0`
- `e200_big + g=4.0`
- `e200_big + g=8.0`
- `e200_big + g=10.0`
- `e200_big + g=14.0`
- `e200_big + g=16.0`
- `e400_big_b128 + g=4.0`

Best by objective:

| Objective | Best run | Best schedule/order | Value |
| --- | --- | --- | ---: |
| posterior_quality | `e200_big + g=16.0` | `every_5 / gauss_newton` | `3.136363` |
| physical_consistency | `e200_big + g=16.0` | `every_step / gauss_newton` | `0.000359` |
| runtime | `baseline` | `none / none` | `0.608586 s` |
| trajectory_stability | `baseline` | `none / none` | `0.0` |

Best inverse-recovery row (minimum `u_error` across all full studies):
- Run: `e200_big + g=16.0`
- Schedule/order: `every_step / first_order`
- `posterior_quality=3.229541`
- `u_error=1.504957`
- `v_error=0.912022`
- `obs_error=0.812562`
- `runtime=1.374069 s`

### Do objectives prefer different schedules/orders?

Yes.  
- Posterior recovery (`u_error`, `posterior_quality`) prefers projected trajectories, especially first-order frequent projection.
- Physical consistency prefers `gauss_newton` with aggressive projection.
- Runtime and trajectory stability prefer `none / none`.

### Full-table comparison vs previous best (`g=10.0` full)

Compared full tables:
- Previous best: `outputs/guidance_big_g10.0/eval_full.json`
- Challengers: `outputs/guidance_big_g14.0/eval_full.json`, `outputs/guidance_big_g16.0/eval_full.json`

Best inverse-recovery row (`u_error`) deltas vs `g=10.0`:
- `g=14.0`:
  - `posterior_quality`: `-4.61%`
  - `u_error`: `-1.91%`
  - `v_error`: `-6.63%`
  - `obs_error`: `-6.98%`
  - `runtime`: `+1.06%`
- `g=16.0`:
  - `posterior_quality`: `-6.75%`
  - `u_error`: `-2.81%`
  - `v_error`: `-9.72%`
  - `obs_error`: `-10.17%`
  - `runtime`: `+4.21%`

Takeaway:
- `g=16.0` is the strongest recovery setting so far.
- `g=14.0` also improves over `g=10.0`, but not as much as `g=16.0`.

## 4. Improvement vs Baseline (Best Inverse Row to Best Inverse Row)

Baseline best inverse row:
- `every_step / first_order`
- `posterior_quality=4.367923`
- `u_error=1.923210`
- `v_error=1.294356`
- `obs_error=1.150357`
- `runtime=0.735697 s`

Best tuned inverse row (`e200_big + g=16.0`):
- `every_step / first_order`
- `posterior_quality=3.229541`
- `u_error=1.504957`
- `v_error=0.912022`
- `obs_error=0.812562`
- `runtime=1.374069 s`

Relative change:
- `posterior_quality`: `-26.06%` (better)
- `u_error`: `-21.75%` (better)
- `v_error`: `-29.54%` (better)
- `obs_error`: `-29.36%` (better)
- `runtime`: `+86.77%` (slower)

Relative to previous best (`g=10.0`, best inverse row):
- `posterior_quality`: `-6.75%`
- `u_error`: `-2.81%`
- `v_error`: `-9.72%`
- `obs_error`: `-10.17%`
- `runtime`: `+4.21%`

Conclusion: posterior recovery improved materially after stronger training/capacity/guidance tuning.

## 5. Step-Count Tuning Decision (`num_steps=100` vs `200`)

Medium checks (`32 x 2`) at `g=8.0` and `g=4.0` both show:
- negligible quality changes (sub-1% scale on key inverse metrics),
- large runtime increase (~2x or more) at `num_steps=200`.

Decision:
- keep `num_steps=100` for the recommended setting.

## 6. Recommended Next Config

Primary recommendation (inverse posterior recovery):
- Checkpoint: `outputs/posterior_projection_e200_big/best.pt`
- Sampling: `--observation-guidance-strength 16.0 --num-steps 100`
- Projection strategy: `every_step / first_order`

Alternative by objective:
- Best physical consistency: `g=16.0` + `every_step / gauss_newton`
- Best runtime / stability: `none / none`

## 7. Commands Run (This Completion Phase)

Validation:
```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Additional full ranking run:
```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e400_big_b128/best.pt \
  --observation-guidance-strength 4.0 \
  --num-samples 128 \
  --num-observation-seeds 3 \
  --device cuda:7 \
  --json-out outputs/guidance_e400bigb128_g4.0/eval_full.json \
  --csv-out outputs/guidance_e400bigb128_g4.0/eval_full.csv
```

Step-count sweep runs:
```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 8.0 \
  --num-steps 200 \
  --num-samples 32 \
  --num-observation-seeds 2 \
  --device cuda:1 \
  --json-out outputs/guidance_big_g8.0_steps200/eval_32x2.json \
  --csv-out outputs/guidance_big_g8.0_steps200/eval_32x2.csv
```

```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 4.0 \
  --num-steps 200 \
  --num-samples 32 \
  --num-observation-seeds 2 \
  --device cuda:7 \
  --json-out outputs/guidance_big_g4.0_steps200/eval_32x2.json \
  --csv-out outputs/guidance_big_g4.0_steps200/eval_32x2.csv
```

Guidance sweep extension + promotion runs:
```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 6.0 \
  --num-samples 32 \
  --num-observation-seeds 2 \
  --device cuda:4 \
  --json-out outputs/guidance_big_g6.0/eval_32x2.json \
  --csv-out outputs/guidance_big_g6.0/eval_32x2.csv
```

```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 10.0 \
  --num-samples 32 \
  --num-observation-seeds 2 \
  --device cuda:2 \
  --json-out outputs/guidance_big_g10.0/eval_32x2.json \
  --csv-out outputs/guidance_big_g10.0/eval_32x2.csv
```

```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 10.0 \
  --num-samples 128 \
  --num-observation-seeds 3 \
  --device cuda:2 \
  --json-out outputs/guidance_big_g10.0/eval_full.json \
  --csv-out outputs/guidance_big_g10.0/eval_full.csv
```

```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 12.0 \
  --num-samples 32 \
  --num-observation-seeds 2 \
  --device cuda:3 \
  --json-out outputs/guidance_big_g12.0/eval_32x2.json \
  --csv-out outputs/guidance_big_g12.0/eval_32x2.csv
```

```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 14.0 \
  --num-samples 32 \
  --num-observation-seeds 2 \
  --device cuda:5 \
  --json-out outputs/guidance_big_g14.0/eval_32x2.json \
  --csv-out outputs/guidance_big_g14.0/eval_32x2.csv
```

```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 16.0 \
  --num-samples 32 \
  --num-observation-seeds 2 \
  --device cuda:6 \
  --json-out outputs/guidance_big_g16.0/eval_32x2.json \
  --csv-out outputs/guidance_big_g16.0/eval_32x2.csv
```

```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 14.0 \
  --num-samples 128 \
  --num-observation-seeds 3 \
  --device cuda:5 \
  --json-out outputs/guidance_big_g14.0/eval_full.json \
  --csv-out outputs/guidance_big_g14.0/eval_full.csv
```

```bash
PYTHONPATH=src .venv/bin/python -u -m posterior_projection.evaluate \
  --checkpoint outputs/posterior_projection_e200_big/best.pt \
  --observation-guidance-strength 16.0 \
  --num-samples 128 \
  --num-observation-seeds 3 \
  --device cuda:6 \
  --json-out outputs/guidance_big_g16.0/eval_full.json \
  --csv-out outputs/guidance_big_g16.0/eval_full.csv
```

GPU note:
- One earlier test run failed with CUDA OOM due shared-cluster occupancy; CPU-forced test run passed.

## 8. Presentation Artifacts (LaTeX + Figures)

Prepared a slide-ready LaTeX summary with exact metric definitions and the final study implications:

- `reports/final_projection_study_presentation.tex`

Generated figure assets and data backing that deck:

- `reports/build_presentation_assets.py`
- `reports/presentation_data.json`
- `reports/figures/guidance_trend_best_u.png`
- `reports/figures/runtime_uerror_tradeoff.png`
- `reports/figures/g16_schedule_tradeoff.png`
- `reports/figures/g16_uerror_physics_tradeoff.png`
- `reports/figures/g16_projection_overhead.png`

These slides report:
- PDE/problem setup and discrete operator definitions.
- Exact implemented metric equations (`posterior_quality`, `physical_consistency`, `runtime`, `trajectory_stability`).
- Full-table winners and objective-specific schedule/order tradeoffs.
- Quantitative baseline vs `g=10` vs `g=16` inverse-recovery comparison.

## 9. Algorithm-to-Parameter Mapping (Clarified)

To avoid ambiguity, the study now explicitly maps method design variables to implementation controls:

- `S` (projection schedule) maps to `projection.schedules` and `study.should_project(...)`.
- `q` (projection order) maps to `order in {first_order, gauss_newton}` and `projection.py`.
- Observation conditioning strength maps to `sampling.observation_guidance_strength`.
- Trajectory discretization depth maps to `sampling.num_steps`.
- End-of-trajectory hard-constraint refinement maps to `projection.final_cleanup*`.
- Prior capacity maps to `model.hidden_channels`, `model.num_fno_layers`, `model.modes`.
- Training strength maps to `training.epochs`, `training.batch_size`.

This mapping is now reflected in both the slide deck and this summary to connect each result directly to its causal knob.

## 10. Expected vs Actual (From Proposal/Papers to This Benchmark)

Expectations from proposal references:

- DiffusionPDE suggests stronger physics/observation guidance should improve inverse recovery.
- PCFM suggests intermediate corrections can improve quality versus final-only correction.
- PCFM also warns very aggressive per-step projection can over-constrain trajectories.
- PCFM runtime note suggests final projection can be a small runtime share.

Observed in this project:

- Stronger guidance and stronger prior materially improved inverse recovery (`u_error`, `v_error`, `obs_error`, `posterior_quality`).
- Intermediate first-order projection (`every_step / first_order`) gave the best `u_error`.
- Gauss-Newton usually improved physical consistency but not `u_error`, with noticeable runtime cost.
- Final-only projection had low trajectory distortion but weaker inverse recovery.
- Final-step-only projection overhead was near the small-overhead expectation; frequent second-order projection had substantially larger overhead.

## 11. Important Metric Caveats (for correct interpretation)

- `trajectory_stability` is measured as deviation from the `none/none` baseline trajectory; therefore `none/none` is always exactly `0` by definition.
- `physical_consistency` is ranked on **pre-cleanup** states. Final cleanup can make `post_ce` tiny without changing the pre-cleanup ranking.
- `posterior_quality` combines relative errors and observation fit; it does not directly include residual norm. Therefore, best posterior-quality and best physical-consistency rows can differ.
