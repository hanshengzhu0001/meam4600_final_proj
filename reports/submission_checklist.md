# Final Project Submission Checklist

## Status vs Initial Plan

- [x] Tests pass (CPU-validated on shared cluster node).
- [x] Dataset generation works (`data/chonkdiff_elliptic_dataset.npz` present).
- [x] Baseline training + evaluation completed.
- [x] Full ranking tables produced (`128 x 3`).
- [x] Stronger runs improved posterior recovery materially over baseline.
- [x] Final summary states best schedule/order by objective.

## Current Best Recommendation

- Checkpoint: `outputs/posterior_projection_e200_big/best.pt`
- Sampling: `--observation-guidance-strength 16.0 --num-steps 100`
- Recovery projection strategy: `every_step / first_order`

## Required Closeout Actions

1. Push latest local commits on `study/projection-sweep`.
2. Preserve output artifacts (JSON/CSV under `outputs/`) for submission package since `outputs/` is gitignored.
3. Submit/report with tracked summary:
   - `reports/final_projection_study_summary.md`

## Push Command

```bash
cd /home/kevinzyz/meam4600_final_proj
git push origin study/projection-sweep
```

## Optional Artifact Archive

```bash
cd /home/kevinzyz/meam4600_final_proj
tar -czf final_outputs_bundle.tgz outputs/guidance_big_g10.0 outputs/guidance_big_g14.0 outputs/guidance_big_g16.0 outputs/posterior_projection_baseline reports/final_projection_study_summary.md
```
