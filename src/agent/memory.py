"""Persistencia de memoria del usuario entre sesiones."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.state import UserMemory

# Directorio por defecto para ficheros de memoria
_MEMORY_DIR = Path(__file__).resolve().parent.parent.parent / "memory"


def _memory_path(user_id: str, base_dir: Path | None = None) -> Path:
    d = base_dir or _MEMORY_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{user_id}.json"


def load_memory(user_id: str = "default", base_dir: Path | None = None) -> UserMemory:
    """Carga la memoria del usuario desde disco. Devuelve vacia si no existe."""
    path = _memory_path(user_id, base_dir)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return UserMemory(
            goals=data.get("goals", []),
            past_alerts=data.get("past_alerts", []),
            savings_tips_given=data.get("savings_tips_given", []),
        )
    return UserMemory(goals=[], past_alerts=[], savings_tips_given=[])


def save_memory(
    memory: UserMemory, user_id: str = "default", base_dir: Path | None = None
) -> Path:
    """Guarda la memoria del usuario en disco. Devuelve la ruta del fichero."""
    path = _memory_path(user_id, base_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(memory), f, ensure_ascii=False, indent=2)
    return path
