from __future__ import annotations

import os
from pathlib import Path


_DOTENV_LOADED = False


def load_dotenv(path: str | Path = ".env") -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_env(name: str, fallback: str = "") -> str:
    load_dotenv()
    return os.environ.get(name, fallback).strip()


def get_env_int(name: str, fallback: int, lower: int, upper: int) -> int:
    raw = get_env(name)
    try:
        parsed = int(raw)
    except ValueError:
        parsed = fallback
    return min(max(parsed, lower), upper)
