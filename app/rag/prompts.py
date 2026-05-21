from pathlib import Path

import yaml

_PROMPTS_PATH = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
_cache: dict[str, str] | None = None


def _load_prompts() -> dict[str, str]:
    global _cache
    if _cache is None:
        with _PROMPTS_PATH.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _cache = data.get("system_prompts", {})
    return _cache


def get_system_prompt(name: str = "default") -> str:
    prompts = _load_prompts()
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' not found. Available: {list(prompts.keys())}")
    return prompts[name]
