from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import JointStateDataset
from .pipeline import PosteriorProjectionPipeline
from .problem import JointPosteriorProblem
from .study import build_default_cases


def _observation_error(v_pred: torch.Tensor, obs_v: torch.Tensor, obs_mask: torch.Tensor) -> float:
    mask = obs_mask.to(v_pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return float(torch.sqrt(((mask * (v_pred - obs_v)) ** 2).sum() / denom).item())


def _nan_aware_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.isnan(array).all():
        return float("nan")
    return float(np.nanmean(array))


def _empirical_covariance(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("samples must be 2D (num_samples, feature_dim)")
    feature_dim = samples.shape[1]
    if samples.shape[0] <= 1:
        return np.zeros((feature_dim, feature_dim), dtype=float)
    return np.cov(samples, rowvar=False)


def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    # Numerical covariance estimates are symmetric up to floating-point noise.
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, a_min=0.0, a_max=None)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T


def _frechet_distance(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
) -> float:
    diff = mean_a - mean_b
    sqrt_cov_a = _matrix_sqrt_psd(cov_a)
    inner = sqrt_cov_a @ cov_b @ sqrt_cov_a
    sqrt_inner = _matrix_sqrt_psd(inner)
    value = float(diff @ diff + np.trace(cov_a + cov_b) - 2.0 * np.trace(sqrt_inner))
    return max(0.0, value)


def _distribution_metrics(
    generated_vectors: list[np.ndarray],
    true_vectors: list[np.ndarray],
) -> tuple[float, float, float]:
    generated = np.asarray(generated_vectors, dtype=float)
    truth = np.asarray(true_vectors, dtype=float)
    if generated.size == 0 or truth.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean_generated = generated.mean(axis=0)
    mean_truth = truth.mean(axis=0)
    std_generated = generated.std(axis=0)
    std_truth = truth.std(axis=0)

    # Use mean-squared definitions so values remain comparable across
    # resolutions/families when we add more PDE benchmarks.
    mmse = float(np.mean((mean_generated - mean_truth) ** 2))
    smse = float(np.mean((std_generated - std_truth) ** 2))

    cov_generated = _empirical_covariance(generated)
    cov_truth = _empirical_covariance(truth)
    fpd = _frechet_distance(mean_generated, cov_generated, mean_truth, cov_truth)
    return mmse, smse, fpd


def evaluate(
    checkpoint_path: str,
    json_out: str | None = None,
    csv_out: str | None = None,
    num_samples: int | None = None,
    num_observation_seeds: int | None = None,
    device: str | None = None,
    observation_guidance_strength: float | None = None,
    num_steps: int | None = None,
    final_cleanup_iterations: int | None = None,
    final_cleanup: bool | None = None,
) -> list[dict[str, float | str]]:
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    pipeline = PosteriorProjectionPipeline.from_checkpoint(checkpoint_path, device=device_obj)
    config = pipeline.config
    if num_samples is not None:
        config.evaluation.num_eval_samples = num_samples
    if num_observation_seeds is not None:
        config.evaluation.num_observation_seeds = num_observation_seeds
    if observation_guidance_strength is not None:
        config.sampling.observation_guidance_strength = observation_guidance_strength
    if num_steps is not None:
        config.sampling.num_steps = num_steps
    if final_cleanup_iterations is not None:
        config.projection.final_cleanup_iterations = final_cleanup_iterations
    if final_cleanup is not None:
        config.projection.final_cleanup = final_cleanup
    dataset = JointStateDataset(
        config.problem.reference_dataset_path,
        split="val",
        observed_fraction=config.posterior.observed_fraction,
        observation_noise_std=config.posterior.observation_noise_std,
        observation_pattern=config.posterior.observation_pattern,
        seed=config.posterior.seed,
        max_samples=config.evaluation.num_eval_samples,
    )
    problem = JointPosteriorProblem(config.problem)
    cases = build_default_cases()

    summaries: list[dict[str, float | str]] = []
    per_case: dict[str, list[dict[str, float]]] = {case.name: [] for case in cases}
    per_case_generated_vectors: dict[str, list[np.ndarray]] = {case.name: [] for case in cases}
    per_case_true_vectors: dict[str, list[np.ndarray]] = {case.name: [] for case in cases}

    for sample_index in range(min(config.evaluation.num_eval_samples, len(dataset))):
        sample = dataset[sample_index]
        u_true = sample["u_phys"].to(torch.float64)
        v_true = sample["v_phys"].to(torch.float64)
        obs_mask = sample["obs_mask"]
        obs_v = sample["obs_v"]

        for obs_seed in range(config.evaluation.num_observation_seeds):
            sample_seed = config.evaluation.sample_seed_base + 10_000 * obs_seed + sample_index
            for case in cases:
                output = pipeline.sample_posterior(
                    obs_mask=obs_mask,
                    obs_v=obs_v,
                    schedule=case.schedule,
                    order=case.order,
                    sample_seed=sample_seed,
                )
                baseline_trajectory = output.baseline_trajectory.to(torch.float64)
                trajectory = output.trajectory.to(torch.float64)
                pre_state = output.pre_cleanup_state_phys
                post_state = output.post_cleanup_state_phys

                traj_deviation = float(
                    ((trajectory - baseline_trajectory) ** 2).sum().item()
                )
                pre_u, pre_v = pre_state[0], pre_state[1]
                post_u, post_v = post_state[0], post_state[1]

                posterior_quality = float(
                    (
                        problem.relative_l2(pre_u.unsqueeze(0), u_true.unsqueeze(0))
                        + problem.relative_l2(pre_v.unsqueeze(0), v_true.unsqueeze(0))
                    ).mean().item()
                ) + _observation_error(
                    pipeline.stats.normalize_v(pre_v.float()),
                    obs_v,
                    obs_mask,
                )

                row = {
                    "posterior_quality": posterior_quality,
                    "physical_consistency": float(problem.residual_norm_from_state(pre_state).item()),
                    "runtime": output.total_runtime_seconds,
                    "trajectory_stability": traj_deviation,
                    "u_error": float(problem.relative_l2(pre_u.unsqueeze(0), u_true.unsqueeze(0)).item()),
                    "v_error": float(problem.relative_l2(pre_v.unsqueeze(0), v_true.unsqueeze(0)).item()),
                    "obs_error": _observation_error(
                        pipeline.stats.normalize_v(pre_v.float()),
                        obs_v,
                        obs_mask,
                    ),
                    "ce_ic": float(problem.ce_ic_from_state(pre_state).item()),
                    "ce_bc": float(problem.ce_bc_from_state(pre_state).item()),
                    "ce_cl": float(problem.ce_cl_from_state(pre_state).item()),
                    "ce_squared": float((problem.residual_from_state(pre_state) ** 2).sum().item()),
                    "projection_time": output.projection_time_seconds,
                    "projection_calls": float(output.projection_calls),
                    "projection_iterations": float(output.projection_iterations),
                    "post_u_error": float(problem.relative_l2(post_u.unsqueeze(0), u_true.unsqueeze(0)).item()),
                    "post_v_error": float(problem.relative_l2(post_v.unsqueeze(0), v_true.unsqueeze(0)).item()),
                    "post_ce": float(problem.residual_norm_from_state(post_state).item()),
                    "post_ce_ic": float(problem.ce_ic_from_state(post_state).item()),
                    "post_ce_bc": float(problem.ce_bc_from_state(post_state).item()),
                    "post_ce_cl": float(problem.ce_cl_from_state(post_state).item()),
                }
                per_case[case.name].append(row)
                per_case_generated_vectors[case.name].append(
                    pre_state.reshape(-1).detach().cpu().numpy()
                )
                per_case_true_vectors[case.name].append(
                    torch.stack([u_true, v_true], dim=0).reshape(-1).detach().cpu().numpy()
                )

    for case in cases:
        values = per_case[case.name]
        summary: dict[str, float | str] = {"schedule": case.schedule, "order": case.order}
        for key in values[0].keys():
            summary[key] = _nan_aware_mean([row[key] for row in values])
        mmse, smse, fpd = _distribution_metrics(
            per_case_generated_vectors[case.name],
            per_case_true_vectors[case.name],
        )
        summary["mmse"] = mmse
        summary["smse"] = smse
        summary["fpd"] = fpd
        summaries.append(summary)

    for metric_name in ["posterior_quality", "physical_consistency", "runtime", "trajectory_stability"]:
        ranking = sorted(summaries, key=lambda row: float(row[metric_name]))
        for rank, row in enumerate(ranking, start=1):
            row[f"rank_{metric_name}"] = float(rank)

    if json_out is not None:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if csv_out is not None:
        out_path = Path(csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--json-out", default="outputs/posterior_projection/eval_summary.json")
    parser.add_argument("--csv-out", default="outputs/posterior_projection/eval_summary.csv")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-observation-seeds", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, or cuda:N")
    parser.add_argument("--observation-guidance-strength", type=float, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--final-cleanup-iterations", type=int, default=None)
    parser.add_argument("--no-final-cleanup", dest="final_cleanup", action="store_false", default=None)
    args = parser.parse_args()

    rows = evaluate(
        args.checkpoint,
        json_out=args.json_out,
        csv_out=args.csv_out,
        num_samples=args.num_samples,
        num_observation_seeds=args.num_observation_seeds,
        device=args.device,
        observation_guidance_strength=args.observation_guidance_strength,
        num_steps=args.num_steps,
        final_cleanup_iterations=args.final_cleanup_iterations,
        final_cleanup=args.final_cleanup,
    )
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
