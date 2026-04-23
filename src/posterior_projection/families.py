from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from chonkdiff.benchmark import NonlinearElliptic1D
from chonkdiff.config import BenchmarkConfig, ForcingConfig

from .config import ProblemConfig


@dataclass(frozen=True, slots=True)
class FamilyDatasetSpec:
    u_key: str
    v_key: str
    x_key: str = "x"


class JointProblemBackend(Protocol):
    nx: int

    def residual(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...

    def jacobian_v(self, v: torch.Tensor) -> torch.Tensor: ...

    def ce_ic_from_state(self, state: torch.Tensor) -> torch.Tensor: ...

    def ce_bc_from_state(self, state: torch.Tensor) -> torch.Tensor: ...

    def ce_cl_from_state(self, state: torch.Tensor) -> torch.Tensor: ...


class NonlinearEllipticBackend:
    def __init__(
        self,
        config: ProblemConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        benchmark_config = BenchmarkConfig(
            nx=config.nx,
            domain_length=config.domain_length,
            kappa=config.kappa,
            forcing=ForcingConfig(
                period_length=config.forcing_period_length,
                lengthscale=config.forcing_lengthscale,
                jitter=config.forcing_jitter,
            ),
        )
        self.nx = config.nx
        self.benchmark = NonlinearElliptic1D(
            benchmark_config,
            device=device,
            dtype=dtype,
        )

    def residual(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.benchmark.residual(u, v)

    def jacobian_v(self, v: torch.Tensor) -> torch.Tensor:
        return self.benchmark.jacobian_matrix(v)

    def ce_ic_from_state(self, state: torch.Tensor) -> torch.Tensor:
        # Static elliptic benchmark has no initial-condition constraint.
        return torch.full((), float("nan"), dtype=state.dtype, device=state.device)

    def ce_bc_from_state(self, state: torch.Tensor) -> torch.Tensor:
        v = state[..., 1, :]
        value = self.benchmark.periodic_bc_violation(v)
        return value.to(dtype=state.dtype, device=state.device)

    def ce_cl_from_state(self, state: torch.Tensor) -> torch.Tensor:
        u = state[..., 0, :]
        v = state[..., 1, :]
        residual = self.residual(u, v)
        return torch.linalg.vector_norm(residual, dim=-1)


def _supported_families() -> tuple[str, ...]:
    return ("nonlinear_elliptic",)


def get_dataset_spec(family: str) -> FamilyDatasetSpec:
    if family == "nonlinear_elliptic":
        return FamilyDatasetSpec(
            u_key="{split}_u",
            v_key="{split}_v",
            x_key="x",
        )
    supported = ", ".join(_supported_families())
    raise ValueError(f"unsupported PDE family '{family}'; supported families: {supported}")


def resolve_split_key(template: str, split: str) -> str:
    return template.format(split=split)


def build_problem_backend(
    config: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> JointProblemBackend:
    if config.family == "nonlinear_elliptic":
        return NonlinearEllipticBackend(config=config, device=device, dtype=dtype)
    supported = ", ".join(_supported_families())
    raise ValueError(f"unsupported PDE family '{config.family}'; supported families: {supported}")
