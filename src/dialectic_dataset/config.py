"""Configuration for the dialectic dataset pipeline.

Loads from YAML config file or uses defaults. All settings are overridable
via CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Config:
    """Top-level pipeline configuration."""

    # Generation
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.8
    max_tokens: int = 4096
    max_concurrent: int = 5
    target_count: Optional[int] = None
    categories: Optional[List[str]] = None

    # Quality
    min_quality: float = 0.72
    require_structural_pass: bool = True
    require_would_train: bool = True

    # Diversity
    max_per_prompt: int = 3
    max_per_category: int = 50
    near_duplicate_threshold: float = 0.6

    # Formatting
    explicit_ratio: float = 0.40
    seed: int = 42

    # Paths
    output_dir: str = "data/generated"
    training_dir: str = "data/training"
    goldset_path: str = "data/goldset/goldset_v1.json"

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Optional[str] = None) -> Config:
        """Load config from file if it exists, otherwise use defaults."""
        if path and Path(path).exists():
            return cls.from_yaml(path)

        # Check default locations
        for default in ["dialectic.yaml", "dialectic.yml", ".dialectic.yaml"]:
            if Path(default).exists():
                return cls.from_yaml(default)

        return cls()

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        data = {k: v for k, v in self.__dict__.items() if v is not None}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
