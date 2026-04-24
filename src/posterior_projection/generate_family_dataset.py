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
        x=torch.linspace(
            0.0,
            float(config.problem.domain_length),
            config.problem.nx,
            dtype=torch.float64,
        ).detach().cpu().numpy(),
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


def _build_heat_equation_dataset(
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

    if config.problem.heat_nu <= 0.0:
        raise ValueError("heat_equation_periodic requires problem.heat_nu > 0")
    if config.problem.heat_time <= 0.0:
        raise ValueError("heat_equation_periodic requires problem.heat_time > 0")

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
    initial_conditions = benchmark.sample_forcing(total, seed=seed).u.to(torch.float64)

    eigvals, eigvecs = torch.linalg.eigh(benchmark.base_operator.to(torch.float64))
    decay = torch.exp(-float(config.problem.heat_nu) * float(config.problem.heat_time) * eigvals)
    forward_operator = eigvecs @ torch.diag(decay) @ eigvecs.T
    solutions = torch.matmul(
        forward_operator.unsqueeze(0),
        initial_conditions.unsqueeze(-1),
    ).squeeze(-1)

    u_array = initial_conditions.detach().cpu().numpy()
    v_array = solutions.detach().cpu().numpy()

    train_u = u_array[:train_size]
    train_v = v_array[:train_size]
    val_u = u_array[train_size:]
    val_v = v_array[train_size:]

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
        heat_nu=np.asarray(config.problem.heat_nu, dtype=np.float32),
        heat_time=np.asarray(config.problem.heat_time, dtype=np.float32),
        nx=np.asarray(config.problem.nx, dtype=np.int32),
    )
    return out_path


def _build_reaction_diffusion_implicit_dataset(
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

    if config.problem.reaction_nu <= 0.0:
        raise ValueError("reaction_diffusion_ic_implicit requires problem.reaction_nu > 0")
    if config.problem.reaction_rho <= 0.0:
        raise ValueError("reaction_diffusion_ic_implicit requires problem.reaction_rho > 0")
    if config.problem.reaction_dt <= 0.0:
        raise ValueError("reaction_diffusion_ic_implicit requires problem.reaction_dt > 0")

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
    raw_v = benchmark.sample_forcing(total, seed=seed).u.to(torch.float64)
    # Keep v in a physically meaningful logistic range while preserving smoothness.
    states_v = 0.5 + 0.35 * torch.tanh(raw_v)

    identity = torch.eye(config.problem.nx, dtype=torch.float64)
    linear_operator = (
        (1.0 - float(config.problem.reaction_dt) * float(config.problem.reaction_rho)) * identity
        + float(config.problem.reaction_dt)
        * float(config.problem.reaction_nu)
        * benchmark.base_operator.to(torch.float64)
    )
    states_u = torch.matmul(
        linear_operator.unsqueeze(0),
        states_v.unsqueeze(-1),
    ).squeeze(-1) + float(config.problem.reaction_dt) * float(config.problem.reaction_rho) * (
        states_v**2
    )

    u_array = states_u.detach().cpu().numpy()
    v_array = states_v.detach().cpu().numpy()

    train_u = u_array[:train_size]
    train_v = v_array[:train_size]
    val_u = u_array[train_size:]
    val_v = v_array[train_size:]

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
        reaction_nu=np.asarray(config.problem.reaction_nu, dtype=np.float32),
        reaction_rho=np.asarray(config.problem.reaction_rho, dtype=np.float32),
        reaction_dt=np.asarray(config.problem.reaction_dt, dtype=np.float32),
        nx=np.asarray(config.problem.nx, dtype=np.int32),
    )
    return out_path


def _build_burgers_implicit_dataset(
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

    if config.problem.burgers_nu <= 0.0:
        raise ValueError("burgers_ic_implicit requires problem.burgers_nu > 0")
    if config.problem.burgers_dt <= 0.0:
        raise ValueError("burgers_ic_implicit requires problem.burgers_dt > 0")

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
    raw_v = benchmark.sample_forcing(total, seed=seed).u.to(torch.float64)
    # Keep advection moderate for stable one-step implicit mapping.
    states_v = 0.4 * torch.tanh(raw_v)

    dx = float(config.problem.domain_length) / float(config.problem.nx)
    derivative = torch.zeros((config.problem.nx, config.problem.nx), dtype=torch.float64)
    for index in range(config.problem.nx):
        derivative[index, (index + 1) % config.problem.nx] = 0.5 / dx
        derivative[index, (index - 1) % config.problem.nx] = -0.5 / dx

    d_x_v = torch.matmul(derivative.unsqueeze(0), states_v.unsqueeze(-1)).squeeze(-1)
    laplace_neg_v = torch.matmul(
        benchmark.base_operator.to(torch.float64).unsqueeze(0),
        states_v.unsqueeze(-1),
    ).squeeze(-1)
    states_u = states_v + float(config.problem.burgers_dt) * (
        states_v * d_x_v + float(config.problem.burgers_nu) * laplace_neg_v
    )

    u_array = states_u.detach().cpu().numpy()
    v_array = states_v.detach().cpu().numpy()

    train_u = u_array[:train_size]
    train_v = v_array[:train_size]
    val_u = u_array[train_size:]
    val_v = v_array[train_size:]

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
        burgers_nu=np.asarray(config.problem.burgers_nu, dtype=np.float32),
        burgers_dt=np.asarray(config.problem.burgers_dt, dtype=np.float32),
        nx=np.asarray(config.problem.nx, dtype=np.int32),
    )
    return out_path


def _build_burgers_bc_dirichlet_dataset(
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

    if config.problem.burgers_nu <= 0.0:
        raise ValueError("burgers_bc_dirichlet requires problem.burgers_nu > 0")
    if config.problem.nx < 3:
        raise ValueError("burgers_bc_dirichlet requires nx >= 3")

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
    raw_v = benchmark.sample_forcing(total, seed=seed).u.to(torch.float64)
    states_v = 0.35 * torch.tanh(raw_v)
    states_v[:, 0] = float(config.problem.burgers_bc_left)
    states_v[:, -1] = float(config.problem.burgers_bc_right)

    dx = float(config.problem.domain_length) / float(config.problem.nx - 1)
    first_derivative = torch.zeros((config.problem.nx, config.problem.nx), dtype=torch.float64)
    second_derivative = torch.zeros((config.problem.nx, config.problem.nx), dtype=torch.float64)
    for index in range(1, config.problem.nx - 1):
        first_derivative[index, index + 1] = 0.5 / dx
        first_derivative[index, index - 1] = -0.5 / dx
        second_derivative[index, index + 1] = 1.0 / (dx * dx)
        second_derivative[index, index] = -2.0 / (dx * dx)
        second_derivative[index, index - 1] = 1.0 / (dx * dx)

    d_x_v = torch.matmul(first_derivative.unsqueeze(0), states_v.unsqueeze(-1)).squeeze(-1)
    d_xx_v = torch.matmul(second_derivative.unsqueeze(0), states_v.unsqueeze(-1)).squeeze(-1)
    states_u = -float(config.problem.burgers_nu) * d_xx_v + states_v * d_x_v
    states_u[:, 0] = float(config.problem.burgers_bc_left)
    states_u[:, -1] = float(config.problem.burgers_bc_right)

    u_array = states_u.detach().cpu().numpy()
    v_array = states_v.detach().cpu().numpy()

    train_u = u_array[:train_size]
    train_v = v_array[:train_size]
    val_u = u_array[train_size:]
    val_v = v_array[train_size:]

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
        burgers_nu=np.asarray(config.problem.burgers_nu, dtype=np.float32),
        burgers_bc_left=np.asarray(config.problem.burgers_bc_left, dtype=np.float32),
        burgers_bc_right=np.asarray(config.problem.burgers_bc_right, dtype=np.float32),
        nx=np.asarray(config.problem.nx, dtype=np.int32),
    )
    return out_path


def _build_navier_stokes_implicit_dataset(
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

    if config.problem.ns_nu <= 0.0:
        raise ValueError("navier_stokes_1d_implicit requires problem.ns_nu > 0")
    if config.problem.ns_dt <= 0.0:
        raise ValueError("navier_stokes_1d_implicit requires problem.ns_dt > 0")

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
    raw_v = benchmark.sample_forcing(total, seed=seed).u.to(torch.float64)
    states_v = 0.35 * torch.tanh(raw_v)

    dx = float(config.problem.domain_length) / float(config.problem.nx)
    derivative = torch.zeros((config.problem.nx, config.problem.nx), dtype=torch.float64)
    for index in range(config.problem.nx):
        derivative[index, (index + 1) % config.problem.nx] = 0.5 / dx
        derivative[index, (index - 1) % config.problem.nx] = -0.5 / dx

    d_x_v = torch.matmul(derivative.unsqueeze(0), states_v.unsqueeze(-1)).squeeze(-1)
    laplace_neg_v = torch.matmul(
        benchmark.base_operator.to(torch.float64).unsqueeze(0),
        states_v.unsqueeze(-1),
    ).squeeze(-1)
    forcing = float(config.problem.ns_forcing_amplitude) * torch.sin(
        2.0 * torch.pi * benchmark.x.to(torch.float64) / float(config.problem.domain_length)
    )
    states_u = states_v + float(config.problem.ns_dt) * (
        float(config.problem.ns_advection_speed) * d_x_v
        + float(config.problem.ns_nu) * laplace_neg_v
        - forcing.unsqueeze(0)
    )

    u_array = states_u.detach().cpu().numpy()
    v_array = states_v.detach().cpu().numpy()

    train_u = u_array[:train_size]
    train_v = v_array[:train_size]
    val_u = u_array[train_size:]
    val_v = v_array[train_size:]

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
        ns_nu=np.asarray(config.problem.ns_nu, dtype=np.float32),
        ns_dt=np.asarray(config.problem.ns_dt, dtype=np.float32),
        ns_advection_speed=np.asarray(config.problem.ns_advection_speed, dtype=np.float32),
        ns_forcing_amplitude=np.asarray(config.problem.ns_forcing_amplitude, dtype=np.float32),
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
    if config.problem.family == "heat_equation_periodic":
        return _build_heat_equation_dataset(
            config=config,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
            out_path=target,
            force=force,
        )
    if config.problem.family == "reaction_diffusion_ic_implicit":
        return _build_reaction_diffusion_implicit_dataset(
            config=config,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
            out_path=target,
            force=force,
        )
    if config.problem.family == "burgers_ic_implicit":
        return _build_burgers_implicit_dataset(
            config=config,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
            out_path=target,
            force=force,
        )
    if config.problem.family == "burgers_bc_dirichlet":
        return _build_burgers_bc_dirichlet_dataset(
            config=config,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
            out_path=target,
            force=force,
        )
    if config.problem.family == "navier_stokes_1d_implicit":
        return _build_navier_stokes_implicit_dataset(
            config=config,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
            out_path=target,
            force=force,
        )
    raise ValueError(
        f"dataset generation not implemented for family '{config.problem.family}' "
        "(implemented: linear_elliptic_helmholtz, heat_equation_periodic, "
        "reaction_diffusion_ic_implicit, burgers_ic_implicit, burgers_bc_dirichlet, "
        "navier_stokes_1d_implicit)"
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
