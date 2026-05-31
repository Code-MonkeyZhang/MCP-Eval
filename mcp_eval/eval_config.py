"""Evaluation-specific configuration module.

Provides configuration loading for evaluation runs, separate from the
interactive agent's Config class.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    """LLM connection configuration for either Agent or Judge."""

    provider: str = "anthropic"
    api_key: str = ""
    api_base: str = ""
    model: str = ""


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    agent_llm: LLMConfig
    judge_llm: LLMConfig
    system_prompt: str = ""
    max_steps: int = 5
    timeout: int = 120

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load evaluation configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            An EvalConfig instance.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If required fields are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Evaluation config file is empty")

        agent_llm = LLMConfig(**data.get("agent_llm", {}))
        judge_llm = LLMConfig(**data.get("judge_llm", {}))

        return cls(
            agent_llm=agent_llm,
            judge_llm=judge_llm,
            system_prompt=data.get("system_prompt", ""),
            max_steps=data.get("max_steps", 5),
            timeout=data.get("timeout", 120),
        )
