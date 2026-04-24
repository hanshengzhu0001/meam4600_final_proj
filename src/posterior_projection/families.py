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


class LinearEllipticHelmholtzBackend:
    """Periodic 1D linear elliptic family: -Delta v + beta v = u."""

    def __init__(
        self,
        config: ProblemConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        benchmark_config = BenchmarkConfig(
            nx=config.nx,
            domain_length=config.domain_length,
            kappa=0.0,
            forcing=ForcingConfig(
                period_length=config.forcing_period_length,
                lengthscale=config.forcing_lengthscale,
                jitter=config.forcing_jitter,
            ),
        )
        self.nx = config.nx
        self.beta = float(config.linear_beta)
        self.benchmark = NonlinearElliptic1D(
            benchmark_config,
            device=device,
            dtype=dtype,
        )

    def residual(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.benchmark.apply_minus_laplacian(v) + self.beta * v - u

    def jacobian_v(self, v: torch.Tensor) -> torch.Tensor:
        base = self.benchmark.base_operator.to(device=v.device, dtype=v.dtype)
        operator = base + self.beta * torch.eye(self.nx, dtype=v.dtype, device=v.device)
        if v.ndim == 1:
            return operator
        return operator.unsqueeze(0).expand(v.shape[0], -1, -1)

    def ce_ic_from_state(self, state: torch.Tensor) -> torch.Tensor:
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


class HeatEquationPeriodicBackend:
    """Periodic heat family via time-horizon operator consistency.

    We model:
        d_t w = nu * Delta w,   w(0) = u,   v = w(T)
    and enforce residual:
        h(u, v) = K_inv v - u = 0
    where K = exp(-nu*T*(-Delta)).
    """

    def __init__(
        self,
        config: ProblemConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        benchmark_config = BenchmarkConfig(
            nx=config.nx,
            domain_length=config.domain_length,
            kappa=0.0,
            forcing=ForcingConfig(
                period_length=config.forcing_period_length,
                lengthscale=config.forcing_lengthscale,
                jitter=config.forcing_jitter,
            ),
        )
        self.nx = config.nx
        self.nu = float(config.heat_nu)
        self.time_horizon = float(config.heat_time)
        if self.nu <= 0.0:
            raise ValueError("heat_equation_periodic requires problem.heat_nu > 0")
        if self.time_horizon <= 0.0:
            raise ValueError("heat_equation_periodic requires problem.heat_time > 0")
        self.benchmark = NonlinearElliptic1D(
            benchmark_config,
            device=device,
            dtype=dtype,
        )

        base = self.benchmark.base_operator.to(device=device, dtype=dtype)
        eigvals, eigvecs = torch.linalg.eigh(base)
        self._eigvals = eigvals
        self._eigvecs = eigvecs
        decay = torch.exp(-self.nu * self.time_horizon * eigvals)
        inv_decay = torch.exp(self.nu * self.time_horizon * eigvals)
        self._forward_operator = eigvecs @ torch.diag(decay) @ eigvecs.T
        self._inverse_operator = eigvecs @ torch.diag(inv_decay) @ eigvecs.T

    def apply_forward(self, u: torch.Tensor) -> torch.Tensor:
        operator = self._forward_operator.to(device=u.device, dtype=u.dtype)
        if u.ndim == 1:
            return operator @ u
        return torch.matmul(operator.unsqueeze(0), u.unsqueeze(-1)).squeeze(-1)

    def residual(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        operator = self._inverse_operator.to(device=v.device, dtype=v.dtype)
        if v.ndim == 1:
            lifted = operator @ v
        else:
            lifted = torch.matmul(operator.unsqueeze(0), v.unsqueeze(-1)).squeeze(-1)
        return lifted - u

    def jacobian_v(self, v: torch.Tensor) -> torch.Tensor:
        operator = self._inverse_operator.to(device=v.device, dtype=v.dtype)
        if v.ndim == 1:
            return operator
        return operator.unsqueeze(0).expand(v.shape[0], -1, -1)

    def ce_ic_from_state(self, state: torch.Tensor) -> torch.Tensor:
        # IC is represented by the recovered u itself in this parameterization.
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


class ReactionDiffusionImplicitICBackend:
    """One-step implicit periodic reaction-diffusion IC family.

    Backward-Euler relation over dt:
        (v - u) / dt = nu * Delta(v) + rho * v * (1 - v)
    with periodic BC, where u is the state at t=0 and v is the state at t=dt.
    We encode this as h(u, v) = 0:
        h(u, v) = v + dt*nu*(-Delta)v - dt*rho*v*(1-v) - u.
    """

    def __init__(
        self,
        config: ProblemConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        benchmark_config = BenchmarkConfig(
            nx=config.nx,
            domain_length=config.domain_length,
            kappa=0.0,
            forcing=ForcingConfig(
                period_length=config.forcing_period_length,
                lengthscale=config.forcing_lengthscale,
                jitter=config.forcing_jitter,
            ),
        )
        self.nx = config.nx
        self.nu = float(config.reaction_nu)
        self.rho = float(config.reaction_rho)
        self.dt = float(config.reaction_dt)
        if self.nu <= 0.0:
            raise ValueError("reaction_diffusion_ic_implicit requires problem.reaction_nu > 0")
        if self.rho <= 0.0:
            raise ValueError("reaction_diffusion_ic_implicit requires problem.reaction_rho > 0")
        if self.dt <= 0.0:
            raise ValueError("reaction_diffusion_ic_implicit requires problem.reaction_dt > 0")

        self.benchmark = NonlinearElliptic1D(
            benchmark_config,
            device=device,
            dtype=dtype,
        )

        laplace_neg = self.benchmark.base_operator.to(device=device, dtype=dtype)
        identity = torch.eye(self.nx, dtype=dtype, device=device)
        self._linear_operator = (
            (1.0 - self.dt * self.rho) * identity + self.dt * self.nu * laplace_neg
        )

    def _apply_linear(self, v: torch.Tensor) -> torch.Tensor:
        operator = self._linear_operator.to(device=v.device, dtype=v.dtype)
        if v.ndim == 1:
            return operator @ v
        return torch.matmul(operator.unsqueeze(0), v.unsqueeze(-1)).squeeze(-1)

    def residual(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self._apply_linear(v) + self.dt * self.rho * (v**2) - u

    def jacobian_v(self, v: torch.Tensor) -> torch.Tensor:
        operator = self._linear_operator.to(device=v.device, dtype=v.dtype)
        if v.ndim == 1:
            return operator + torch.diag(2.0 * self.dt * self.rho * v)
        diagonal = torch.diag_embed(2.0 * self.dt * self.rho * v)
        return operator.unsqueeze(0).expand(v.shape[0], -1, -1) + diagonal

    def ce_ic_from_state(self, state: torch.Tensor) -> torch.Tensor:
        # In this inverse setting, u itself is the recovered IC field.
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


class BurgersICImplicitBackend:
    """One-step implicit periodic viscous Burgers IC family.

    Backward-Euler relation over dt:
        (v - u) / dt + v * d_x(v) = nu * d_xx(v)
    where u is t=0 and v is t=dt.
    Rearranged residual:
        h(u, v) = v + dt * (v * d_x(v) + nu * (-Delta)v) - u.
    """

    def __init__(
        self,
        config: ProblemConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        benchmark_config = BenchmarkConfig(
            nx=config.nx,
            domain_length=config.domain_length,
            kappa=0.0,
            forcing=ForcingConfig(
                period_length=config.forcing_period_length,
                lengthscale=config.forcing_lengthscale,
                jitter=config.forcing_jitter,
            ),
        )
        self.nx = config.nx
        self.nu = float(config.burgers_nu)
        self.dt = float(config.burgers_dt)
        if self.nu <= 0.0:
            raise ValueError("burgers_ic_implicit requires problem.burgers_nu > 0")
        if self.dt <= 0.0:
            raise ValueError("burgers_ic_implicit requires problem.burgers_dt > 0")

        self.benchmark = NonlinearElliptic1D(
            benchmark_config,
            device=device,
            dtype=dtype,
        )
        dx = float(config.domain_length) / float(config.nx)
        derivative = torch.zeros((self.nx, self.nx), dtype=dtype, device=device)
        for index in range(self.nx):
            derivative[index, (index + 1) % self.nx] = 0.5 / dx
            derivative[index, (index - 1) % self.nx] = -0.5 / dx
        self._first_derivative = derivative
        self._laplace_neg = self.benchmark.base_operator.to(device=device, dtype=dtype)
        self._identity = torch.eye(self.nx, dtype=dtype, device=device)

    def _apply_operator(self, operator: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        op = operator.to(device=field.device, dtype=field.dtype)
        if field.ndim == 1:
            return op @ field
        return torch.matmul(op.unsqueeze(0), field.unsqueeze(-1)).squeeze(-1)

    def residual(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d_x_v = self._apply_operator(self._first_derivative, v)
        laplace_neg_v = self._apply_operator(self._laplace_neg, v)
        return v + self.dt * (v * d_x_v + self.nu * laplace_neg_v) - u

    def jacobian_v(self, v: torch.Tensor) -> torch.Tensor:
        derivative = self._first_derivative.to(device=v.device, dtype=v.dtype)
        laplace_neg = self._laplace_neg.to(device=v.device, dtype=v.dtype)
        identity = self._identity.to(device=v.device, dtype=v.dtype)
        d_x_v = self._apply_operator(derivative, v)

        if v.ndim == 1:
            advection_jacobian = torch.diag(d_x_v) + torch.diag(v) @ derivative
            return identity + self.dt * (advection_jacobian + self.nu * laplace_neg)

        batch = v.shape[0]
        diag_dx = torch.diag_embed(d_x_v)
        diag_v = torch.diag_embed(v)
        derivative_batch = derivative.unsqueeze(0).expand(batch, -1, -1)
        advection_jacobian = diag_dx + torch.matmul(diag_v, derivative_batch)
        return identity.unsqueeze(0).expand(batch, -1, -1) + self.dt * (
            advection_jacobian + self.nu * laplace_neg.unsqueeze(0).expand(batch, -1, -1)
        )

    def ce_ic_from_state(self, state: torch.Tensor) -> torch.Tensor:
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
    return (
        "nonlinear_elliptic",
        "linear_elliptic_helmholtz",
        "heat_equation_periodic",
        "reaction_diffusion_ic_implicit",
        "burgers_ic_implicit",
    )


def get_dataset_spec(family: str) -> FamilyDatasetSpec:
    if family == "nonlinear_elliptic":
        return FamilyDatasetSpec(
            u_key="{split}_u",
            v_key="{split}_v",
            x_key="x",
        )
    if family == "linear_elliptic_helmholtz":
        return FamilyDatasetSpec(
            u_key="{split}_u",
            v_key="{split}_v",
            x_key="x",
        )
    if family == "heat_equation_periodic":
        return FamilyDatasetSpec(
            u_key="{split}_u",
            v_key="{split}_v",
            x_key="x",
        )
    if family == "reaction_diffusion_ic_implicit":
        return FamilyDatasetSpec(
            u_key="{split}_u",
            v_key="{split}_v",
            x_key="x",
        )
    if family == "burgers_ic_implicit":
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
    if config.family == "linear_elliptic_helmholtz":
        return LinearEllipticHelmholtzBackend(config=config, device=device, dtype=dtype)
    if config.family == "heat_equation_periodic":
        return HeatEquationPeriodicBackend(config=config, device=device, dtype=dtype)
    if config.family == "reaction_diffusion_ic_implicit":
        return ReactionDiffusionImplicitICBackend(config=config, device=device, dtype=dtype)
    if config.family == "burgers_ic_implicit":
        return BurgersICImplicitBackend(config=config, device=device, dtype=dtype)
    supported = ", ".join(_supported_families())
    raise ValueError(f"unsupported PDE family '{config.family}'; supported families: {supported}")
