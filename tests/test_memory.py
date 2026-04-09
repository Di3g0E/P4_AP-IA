"""Tests para la memoria persistente del agente."""

import json
import pytest
from pathlib import Path

from src.agent.memory import load_memory, save_memory
from src.agent.state import UserMemory


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Directorio temporal para ficheros de memoria."""
    return tmp_path / "memory"


class TestMemory:
    def test_load_memory_empty(self, tmp_memory_dir):
        memory = load_memory("test_user", base_dir=tmp_memory_dir)
        assert memory["goals"] == []
        assert memory["past_alerts"] == []
        assert memory["savings_tips_given"] == []

    def test_save_and_load_roundtrip(self, tmp_memory_dir):
        memory = UserMemory(
            goals=[{
                "category": "Ocio",
                "limit": 150.0,
                "deadline": "2026-06",
                "created_at": "2026-04-08",
            }],
            past_alerts=["alerta_ocio_2026-04_80"],
            savings_tips_given=["reducir suscripciones"],
        )
        path = save_memory(memory, "test_user", base_dir=tmp_memory_dir)
        assert path.exists()

        loaded = load_memory("test_user", base_dir=tmp_memory_dir)
        assert len(loaded["goals"]) == 1
        assert loaded["goals"][0]["category"] == "Ocio"
        assert loaded["goals"][0]["limit"] == 150.0
        assert loaded["past_alerts"] == ["alerta_ocio_2026-04_80"]

    def test_save_creates_directory(self, tmp_path):
        nested_dir = tmp_path / "a" / "b" / "c"
        memory = UserMemory(goals=[], past_alerts=[], savings_tips_given=[])
        path = save_memory(memory, "user1", base_dir=nested_dir)
        assert path.exists()

    def test_save_overwrites_existing(self, tmp_memory_dir):
        m1 = UserMemory(goals=[], past_alerts=["old_alert"], savings_tips_given=[])
        save_memory(m1, "user1", base_dir=tmp_memory_dir)

        m2 = UserMemory(goals=[], past_alerts=["new_alert"], savings_tips_given=[])
        save_memory(m2, "user1", base_dir=tmp_memory_dir)

        loaded = load_memory("user1", base_dir=tmp_memory_dir)
        assert loaded["past_alerts"] == ["new_alert"]

    def test_memory_file_is_valid_json(self, tmp_memory_dir):
        memory = UserMemory(
            goals=[{"category": "Test", "limit": 100, "deadline": "2026-12", "created_at": "2026-01-01"}],
            past_alerts=[],
            savings_tips_given=[],
        )
        path = save_memory(memory, "json_test", base_dir=tmp_memory_dir)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "goals" in data
        assert isinstance(data["goals"], list)

    def test_multiple_users_independent(self, tmp_memory_dir):
        m1 = UserMemory(goals=[], past_alerts=["alert_a"], savings_tips_given=[])
        m2 = UserMemory(goals=[], past_alerts=["alert_b"], savings_tips_given=[])
        save_memory(m1, "alice", base_dir=tmp_memory_dir)
        save_memory(m2, "bob", base_dir=tmp_memory_dir)

        alice = load_memory("alice", base_dir=tmp_memory_dir)
        bob = load_memory("bob", base_dir=tmp_memory_dir)
        assert alice["past_alerts"] == ["alert_a"]
        assert bob["past_alerts"] == ["alert_b"]
