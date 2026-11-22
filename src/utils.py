"""Shared offline helpers: config loading and issues.log handling (no network)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | str = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def root_path(rel: str) -> Path:
    return ROOT / rel


def write_issues(cfg: dict, stage: str, lines: list[str]) -> Path:
    """Replace all lines tagged [<stage>] in issues.log with `lines` (idempotent reruns)."""
    path = root_path(cfg["paths"]["issues_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"[{stage}]"
    kept = []
    if path.exists():
        kept = [l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith(tag)]
    new = [f"{tag}{l}" if l.startswith("[") else f"{tag} {l}" for l in lines]
    path.write_text("\n".join(kept + new) + "\n", encoding="utf-8")
    return path
