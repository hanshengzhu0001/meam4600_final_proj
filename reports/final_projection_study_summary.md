# Final Project Study Summary (Tracked Report)

Date: 2026-04-21  
Repo: `meam4600_final_proj`  
Branch: `study/projection-sweep`  
Base commit used for study: `a0b111e`  
Current HEAD at report time: `0862d3ec0eade63636ac2b3834388f31c46fa0c9`

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
- `outputs/guidance_e400bigb128_g4.0/eval_full.json`
- `outputs/guidance_e400bigb128_g4.0/eval_full.csv`

Medium `32 x 2` step-count checks:
- `outputs/guidance_big_g8.0_steps200/eval_32x2.json`
- `outputs/guidance_big_g8.0_steps200/eval_32x2.csv`
- `outputs/guidance_big_g4.0_steps200/eval_32x2.json`
- `outputs/guidance_big_g4.0_steps200/eval_32x2.csv`

Note: `outputs/` is gitignored, so these artifacts are local cluster outputs by design.

## 3. Objective Winners (Full Study)

Compared full runs:
- baseline (`80ep`)
- `e200_big + g=2.0`
- `e200_big + g=4.0`
- `e200_big + g=8.0`
- `e400_big_b128 + g=4.0`

Best by objective:

| Objective | Best run | Best schedule/order | Value |
| --- | --- | --- | ---: |
| posterior_quality | `e200_big + g=8.0` | `every_5 / gauss_newton` | `3.539688` |
| physical_consistency | `e200_big + g=8.0` | `every_step / gauss_newton` | `0.000367` |
| runtime | `baseline` | `none / none` | `0.608586 s` |
| trajectory_stability | `baseline` | `none / none` | `0.0` |

Best inverse-recovery row (minimum `u_error` across all full studies):
- Run: `e200_big + g=8.0`
- Schedule/order: `every_step / first_order`
- `posterior_quality=3.550816`
- `u_error=1.564131`
- `v_error=1.047024`
- `obs_error=0.939660`
- `runtime=1.346538 s`

### Do objectives prefer different schedules/orders?

Yes.  
- Posterior recovery (`u_error`, `posterior_quality`) prefers projected trajectories, especially first-order frequent projection.
- Physical consistency prefers `gauss_newton` with aggressive projection.
- Runtime and trajectory stability prefer `none / none`.

## 4. Improvement vs Baseline (Best Inverse Row to Best Inverse Row)

Baseline best inverse row:
- `every_step / first_order`
- `posterior_quality=4.367923`
- `u_error=1.923210`
- `v_error=1.294356`
- `obs_error=1.150357`
- `runtime=0.735697 s`

Best tuned inverse row (`e200_big + g=8.0`):
- `every_step / first_order`
- `posterior_quality=3.550816`
- `u_error=1.564131`
- `v_error=1.047024`
- `obs_error=0.939660`
- `runtime=1.346538 s`

Relative change:
- `posterior_quality`: `-18.71%` (better)
- `u_error`: `-18.67%` (better)
- `v_error`: `-19.11%` (better)
- `obs_error`: `-18.32%` (better)
- `runtime`: `+83.03%` (slower)

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
- Sampling: `--observation-guidance-strength 8.0 --num-steps 100`
- Projection strategy: `every_step / first_order`

Alternative by objective:
- Best physical consistency: `every_step / gauss_newton`
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

GPU note:
- One earlier test run failed with CUDA OOM due shared-cluster occupancy; CPU-forced test run passed.
