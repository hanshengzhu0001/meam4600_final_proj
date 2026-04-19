"""Nonlinear elliptic benchmark backend used by posterior_projection."""

from .benchmark import NonlinearElliptic1D
from .config import BenchmarkConfig, DatasetConfig, ExperimentConfig, ForcingConfig, OracleConfig, load_config
from .dataset import NormalizationStats, OracleSolutionDataset, generate_oracle_dataset
from .oracle import LMResult, lm_project

__all__ = [
    "BenchmarkConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "ForcingConfig",
    "LMResult",
    "NonlinearElliptic1D",
    "NormalizationStats",
    "OracleConfig",
    "OracleSolutionDataset",
    "generate_oracle_dataset",
    "lm_project",
    "load_config",
]
