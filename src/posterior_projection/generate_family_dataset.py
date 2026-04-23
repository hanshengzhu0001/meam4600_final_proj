from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from chonkdiff.benchmark import NonlinearElliptic1D
from chonkdiff.config import BenchmarkConfig, ForcingConfig

from .config import ExperimentConfig, load_config


def _build_linear_elliptic_dataset(
    config: ExperimentConfig,
    train_size: int,
    val_size: int,
    seed: int,
    out_path: Path,
    force: bool,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return out_path

    if config.problem.linear_beta <= 0.0:
        raise ValueError("linear_elliptic_helmholtz requires problem.linear_beta > 0")

    benchmark = NonlinearElliptic1D(
        BenchmarkConfig(
            nx=config.problem.nx,
            domain_length=config.problem.domain_length,
            kappa=0.0,
            forcing=ForcingConfig(
                period_length=config.problem.forcing_period_length,
                lengthscale=config.problem.forcing_lengthscale,
                jitter=config.problem.forcing_jitter,
            ),
        ),
        dtype=torch.float64,
    )

    total = train_size + val_size
    forcings = benchmark.sample_forcing(total, seed=seed).u.to(torch.float64)

    operator = benchmark.base_operator + float(config.problem.linear_beta) * torch.eye(
        config.problem.nx,
        dtype=torch.float64,
    )
    rhs = forcings.unsqueeze(-1)
    matrix = operator.unsqueeze(0).expand(total, -1, -1)
    solutions = torch.linalg.solve(matrix, rhs).squeeze(-1)

    forcings_array = forcings.detach().cpu().numpy()
    solutions_array = solutions.detach().cpu().numpy()

    train_u = forcings_array[:train_size]
    train_v = solutions_array[:train_size]
    val_u = forcings_array[train_size:]
    val_v = solutions_array[train_size:]

    u_mean = float(train_u.mean())
    u_std = float(train_u.std() + 1.0e-6)
    v_mean = float(train_v.mean())
    v_std = float(train_v.std() + 1.0e-6)

    np.savez_compressed(
        out_path,
        family=np.asarray(config.problem.family),
        equation=np.asarray(config.problem.equation),
        x=benchmark.x.detach().cpu().numpy(),
        train_u=train_u,
        train_v=train_v,
        val_u=val_u,
        val_v=val_v,
        u_mean=np.asarray(u_mean, dtype=np.float32),
        u_std=np.asarray(u_std, dtype=np.float32),
        v_mean=np.asarray(v_mean, dtype=np.float32),
        v_std=np.asarray(v_std, dtype=np.float32),
        linear_beta=np.asarray(config.problem.linear_beta, dtype=np.float32),
        nx=np.asarray(config.problem.nx, dtype=np.int32),
    )
    return out_path


def generate_family_dataset(
    config: ExperimentConfig,
    train_size: int,
    val_size: int,
    seed: int,
    out_path: Path | None = None,
    force: bool = False,
) -> Path:
    target = out_path or Path(config.problem.reference_dataset_path)
    if config.problem.family == "linear_elliptic_helmholtz":
        return _build_linear_elliptic_dataset(
            config=config,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
            out_path=target,
            force=force,
        )
    raise ValueError(
        f"dataset generation not implemented for family '{config.problem.family}' "
        "(implemented: linear_elliptic_helmholtz)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/posterior_projection_linear_elliptic.yaml")
    parser.add_argument("--train-size", type=int, default=896)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    target = Path(args.out_path) if args.out_path is not None else None
    out_path = generate_family_dataset(
        config=config,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        out_path=target,
        force=args.force,
    )
    print(
        f"Generated family dataset: family={config.problem.family} "
        f"train_size={args.train_size} val_size={args.val_size} path={out_path}"
    )


if __name__ == "__main__":
    main()
