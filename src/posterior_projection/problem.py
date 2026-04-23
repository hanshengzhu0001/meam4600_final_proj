from __future__ import annotations

import torch

from .config import ProblemConfig
from .families import JointProblemBackend, build_problem_backend


class JointPosteriorProblem:
    """Joint (u, v) posterior problem wrapper with family-specific backends."""

    def __init__(
        self,
        config: ProblemConfig,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.backend: JointProblemBackend = build_problem_backend(
            config=config,
            device=self.device,
            dtype=dtype,
        )
        self.nx = self.backend.nx

    def split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return state[..., 0, :], state[..., 1, :]

    def combine_state(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.stack([u, v], dim=-2)

    def residual(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.backend.residual(u, v)

    def residual_from_state(self, state: torch.Tensor) -> torch.Tensor:
        u, v = self.split_state(state)
        return self.residual(u, v)

    def residual_norm_from_state(self, state: torch.Tensor) -> torch.Tensor:
        residual = self.residual_from_state(state)
        return torch.linalg.vector_norm(residual, dim=-1)

    def ce_ic_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.backend.ce_ic_from_state(state)

    def ce_bc_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.backend.ce_bc_from_state(state)

    def ce_cl_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.backend.ce_cl_from_state(state)

    def observation_loss(
        self,
        state: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_v: torch.Tensor,
    ) -> torch.Tensor:
        _, v = self.split_state(state)
        normalizer = obs_mask.sum(dim=-1).clamp_min(1.0)
        return ((obs_mask * (v - obs_v)) ** 2).sum(dim=-1) / normalizer

    def joint_jacobian(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        jacobian_v = self.backend.jacobian_v(v)
        if v.ndim == 1:
            neg_identity = -torch.eye(self.nx, dtype=v.dtype, device=v.device)
            return torch.cat([neg_identity, jacobian_v], dim=-1)
        batch = v.shape[0]
        neg_identity = (
            -torch.eye(self.nx, dtype=v.dtype, device=v.device)
            .unsqueeze(0)
            .expand(batch, -1, -1)
        )
        return torch.cat([neg_identity, jacobian_v], dim=-1)

    def joint_jacobian_from_state(self, state: torch.Tensor) -> torch.Tensor:
        u, v = self.split_state(state)
        return self.joint_jacobian(u, v)

    def relative_l2(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = torch.linalg.vector_norm(prediction - target, dim=-1)
        denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(1.0e-12)
        return numerator / denominator
