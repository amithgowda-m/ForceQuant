"""Shared offline helpers: config loading and issues.log handling (no network)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | str = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dotenv(path: Path | str = ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, optional quotes; never overrides real env vars."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key.strip(), val)


def sec_user_agent(cfg: dict, dotenv: Path | str = ROOT / ".env") -> str:
    """SEC_USER_AGENT from the environment (or .env); config.yaml value is only a fallback."""
    load_dotenv(dotenv)
    return os.environ.get("SEC_USER_AGENT") or cfg.get("sec_user_agent", "")


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
