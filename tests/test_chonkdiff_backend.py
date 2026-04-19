from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from chonkdiff.benchmark import NonlinearElliptic1D
from chonkdiff.config import ExperimentConfig
from chonkdiff.dataset import generate_oracle_dataset
from chonkdiff.oracle import lm_project


class ChonkDiffBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig()
        self.config.benchmark.dataset.train_size = 4
        self.config.benchmark.dataset.val_size = 2
        self.config.benchmark.dataset.out_path = str(
            Path(tempfile.gettempdir()) / "meam4600_chonkdiff_backend_test_dataset.npz"
        )
        self.benchmark = NonlinearElliptic1D(self.config.benchmark)

    def test_minus_laplacian_annihilates_constants(self) -> None:
        values = torch.ones(self.config.benchmark.nx, dtype=torch.float64)
        laplacian = self.benchmark.apply_minus_laplacian(values)
        self.assertTrue(torch.allclose(laplacian, torch.zeros_like(laplacian), atol=1.0e-10))

    def test_jacobian_matches_finite_difference(self) -> None:
        forcing = self.benchmark.sample_forcing(1, seed=7).u.squeeze(0)
        state = torch.randn(self.config.benchmark.nx, dtype=torch.float64) * 0.05
        direction = torch.randn(self.config.benchmark.nx, dtype=torch.float64)
        direction = direction / torch.linalg.vector_norm(direction)
        epsilon = 1.0e-6

        finite_difference = (
            self.benchmark.residual(forcing, state + epsilon * direction)
            - self.benchmark.residual(forcing, state - epsilon * direction)
        ) / (2.0 * epsilon)
        jacobian_vector = self.benchmark.jacobian_matrix(state) @ direction
        self.assertTrue(torch.allclose(finite_difference, jacobian_vector, atol=1.0e-5, rtol=1.0e-4))

    def test_projector_reduces_residual(self) -> None:
        forcing = self.benchmark.sample_forcing(1, seed=11).u.squeeze(0)
        result = lm_project(
            self.benchmark,
            forcing,
            torch.zeros_like(forcing),
            self.config.oracle,
            max_iterations=6,
        )
        self.assertGreater(result.residual_history[0], result.residual_history[-1])

    def test_dataset_generation(self) -> None:
        dataset_path = generate_oracle_dataset(self.config, force=True)
        self.assertTrue(Path(dataset_path).exists())


if __name__ == "__main__":
    unittest.main()
