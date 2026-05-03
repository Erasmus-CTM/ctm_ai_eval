import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_serializer, field_validator

from ctm_ai_eval.rag.config import ChunkerConfig
from ctm_ai_eval.utils.path_util import path_collapse_user


class ChatTargetConfig(BaseModel):
    model: str
    system_prompt_id: str
    user_template_id: str
    think: bool = False  # Non-standard, but ollama supports it (?)
    temperature: float = 0.0
    max_tokens: int | None = 512
    use_rag: bool = False


class QaRagConfig(BaseModel):
    """How to setup rag for use in a full pipeline."""

    corpus_path: Path
    embedding_model: str
    chunker: ChunkerConfig

    @field_validator("corpus_path")
    def validate_frame_dir(cls, p: Any):
        """Expand user path"""
        path = Path(p).expanduser()
        return path

    @field_serializer("corpus_path")
    def serialize_frame_dir(self, path: Path) -> str:
        """Serialize the path in a portable format."""
        return str(path_collapse_user(path))


class QaExperimentCfg(BaseModel):
    """Setup for the full QA-experiment."""

    dataset_path: Path = Field(default=Path("./assets/data/general_qa_python.json"))
    prompt_dir: Path = Field(default=Path("./assets/prompts/"))

    targets: list[ChatTargetConfig]
    # a single rag config can be used by all targets.
    rag: QaRagConfig | None = None


def load_qa_config(path: str | None) -> QaExperimentCfg:
    if path is None:
        path = "config_qa.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return QaExperimentCfg.model_validate(raw)
