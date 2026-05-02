import tomllib
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, field_validator

from ctm_ai_eval.utils.path_util import path_collapse_user


class DatasetConfig(BaseModel):
    """For needle extraction and haystack experiment."""

    corpus_path: Path
    max_needles: int

    @field_validator("corpus_path")
    def validate_frame_dir(cls, p: Any):
        """Expand user path"""
        path = Path(p).expanduser()
        return path

    @field_serializer("corpus_path")
    def serialize_frame_dir(self, path: Path) -> str:
        """Serialize the path in a portable format."""
        return str(path_collapse_user(path))


class ChunkerConfig(BaseModel):
    """Should work for most chunkers"""

    type: Literal["word", "sentence", "markdown"]
    length: int
    overlap: int


@dataclass
class EmbeddingConfig:
    models: list[str]


@dataclass
class ExperimentConfig:
    mode: Literal["verbatim", "paraphrase", "query"]
    top_k: int


class RetrievalTargetsCfg(BaseModel):
    embedders: list[str]


class HaystackMetricCfg(BaseModel):
    # which top-k values to compute recall for
    k_vals: tuple[int, ...] = (1, 5, 10)


@dataclass
class BasicLlmCfg:
    model: str
    temperature: float
    think: bool


class HaystackExperimentConfig(BaseModel):
    dataset: DatasetConfig

    needle_llm: BasicLlmCfg
    targets: RetrievalTargetsCfg
    # optionally specify metric details
    metrics: HaystackMetricCfg = Field(default_factory=HaystackMetricCfg)


def load_haystack_config(path: str = "config_haystack.toml") -> HaystackExperimentConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return HaystackExperimentConfig.model_validate(raw)
