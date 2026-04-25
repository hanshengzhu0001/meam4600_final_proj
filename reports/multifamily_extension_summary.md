# Multi-Family Extension Summary

Date: 2026-04-24

## Baseline Family Sweeps Completed

All of the following families completed full guidance sweeps (`g=0.25,1,2,4,8,12,16`) with the same projection schedule/order table:

- `linear_elliptic_helmholtz`
- `heat_equation_periodic`
- `reaction_diffusion_ic_implicit`
- `burgers_ic_implicit`
- `burgers_bc_dirichlet`
- `navier_stokes_1d_implicit`

## Objective Winners by Family (Baseline Checkpoint)

| Family | posterior_quality winner | physical_consistency winner | runtime winner | trajectory_stability winner |
| --- | --- | --- | --- | --- |
| linear_elliptic_helmholtz | `g16, every_5/gauss_newton` (`3.4856`) | `g8, every_step/gauss_newton` (`9.76e-4`) | `g1, none/none` (`0.6405s`) | `g0.25, none/none` (`0.0`) |
| heat_equation_periodic | `g16, every_5/first_order` (`3.1180`) | `g2, every_step/gauss_newton` (`0.3760`) | `g16, none/none` (`0.6774s`) | `g0.25, none/none` (`0.0`) |
| reaction_diffusion_ic_implicit | `g16, every_5/first_order` (`1.6141`) | `g16, adaptive_residual/gauss_newton` (`~1e-6`) | `g2, none/none` (`0.6471s`) | `g0.25, none/none` (`0.0`) |
| burgers_ic_implicit | `g16, every_2/first_order` (`2.7776`) | `g12, every_step/gauss_newton` (`~1e-7`) | `g1, none/none` (`0.6602s`) | `g0.25, none/none` (`0.0`) |
| burgers_bc_dirichlet | `g16, none/none` (`3.4434`) | `g12, final_only/gauss_newton` (`4.0e-6`) | `g12, none/none` (`0.6491s`) | `g0.25, none/none` (`0.0`) |
| navier_stokes_1d_implicit | `g16, every_2/gauss_newton` (`2.8090`) | `g16, every_step/gauss_newton` (`~1e-7`) | `g4, none/none` (`0.6419s`) | `g0.25, none/none` (`0.0`) |

## Guidance Benefit (Best-u Row, `g16` vs `g0.25`)

| Family | u_error | v_error | obs_error | posterior_quality | runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_elliptic_helmholtz | `-18.73%` | `-28.59%` | `-38.09%` | `-26.79%` | `+19.29%` |
| heat_equation_periodic | `-25.20%` | `-26.31%` | `-41.32%` | `-29.33%` | `-3.18%` |
| reaction_diffusion_ic_implicit | `-29.84%` | `-31.38%` | `-42.28%` | `-35.81%` | `+2.31%` |
| burgers_ic_implicit | `-26.12%` | `-27.81%` | `-41.88%` | `-31.18%` | `-1.74%` |
| burgers_bc_dirichlet | `-9.03%` | `-25.43%` | `-34.70%` | `-21.63%` | `+0.27%` |
| navier_stokes_1d_implicit | `-24.11%` | `-28.06%` | `-38.12%` | `-29.56%` | `+18.49%` |

## Stronger-Training Checks

- Reaction-diffusion stronger checkpoint (`e200_b128`) completed full sweep.
  - At matched strongest guidance (`g16`, best-u row): quality was worse than baseline (`u_error +0.21%`, `posterior_quality +5.51%`), though runtime was slightly faster (`-3.46%`).
  - Decision: keep baseline checkpoint as recommended for this family.

- Burgers stronger checkpoint (`e200_b128`) completed matched medium probe (`g16, 32x2`).
  - Compared to matched baseline medium (`g16, 32x2`): worse across key recovery metrics (`u_error +2.62%`, `posterior_quality +2.44%`) and slightly slower runtime (`+0.91%`).
  - Decision: no full promotion for this stronger checkpoint.

- Burgers-BC stronger checkpoint (`e200_b128`) completed matched medium probe (`g16, 32x2`).
  - Compared to matched baseline medium (`g16, 32x2`): `u_error` was worse (`+4.88%`), but `v_error` (`-5.28%`), `obs_error` (`-6.94%`), and `posterior_quality` (`-0.64%`) improved, with faster runtime (`-20.97%`).
  - Decision: keep baseline as default for strict inverse-target `u_error`, but keep stronger checkpoint as a runtime-favoring alternative.

- Navier-Stokes stronger checkpoint (`e200_b128`) completed matched medium probe (`g16, 32x2`).
  - Compared to matched baseline medium (`g16, 32x2`): quality degraded across key metrics (`u_error +6.18%`, `v_error +13.77%`, `obs_error +15.67%`, `posterior_quality +11.33%`) while runtime improved (`-9.12%`).
  - Decision: no full promotion; keep baseline as recommended checkpoint.

## Cross-Family Pattern

- Across all tested families, stronger observation guidance consistently improves inverse recovery (`u_error`, `v_error`, `obs_error`, `posterior_quality`).
- `none/none` remains the stability/runtime anchor case.
- Physical-consistency winners often favor `gauss_newton`, while pure recovery winners often favor `first_order` with frequent projection.
- Burgers-BC is the main outlier in `u_error` sensitivity (smaller `g16` gain than other families), but still follows the same monotonic improvement trend.
- Stronger training is not universally beneficial: several families show better runtime but worse reconstruction quality, so checkpoint selection should remain family-specific.
