"""Tests del nodo de resumen del agente LangGraph."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from src.agent.graph import build_graph
from src.utils.config import KEEP_RECENT_MESSAGES, SUMMARIZATION_THRESHOLD


@pytest.fixture
def sample_df():
    data = {
        "Date": ["01/01/2026", "15/01/2026"],
        "Description": ["Test1", "Test2"],
        "Amount": ["10,00€", "20,00€"],
        "Type": ["Expenses", "Income"],
        "Area": ["Alimentacion", "Salario"],
    }
    df = pd.DataFrame(data)
    df["Amount_clean"] = [10.0, 20.0]
    df["Date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Year"] = df["Date_parsed"].dt.year
    df["Month"] = df["Date_parsed"].dt.month
    df["YearMonth"] = df["Date_parsed"].dt.to_period("M")
    return df


@pytest.fixture
def mock_chatgroq():
    """Mock de ChatGroq para no hacer llamadas reales a la API."""
    with patch("src.agent.graph.ChatGroq") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.bind_tools.return_value = mock_instance
        mock_instance.invoke.return_value = AIMessage(content="Resumen mockeado")
        mock_cls.return_value = mock_instance
        yield mock_cls


class TestGraphStructure:
    def test_graph_compiles_with_summarize_node(self, sample_df, mock_chatgroq):
        graph = build_graph(
            df=sample_df, api_key="fake_key", model="fake_model", user_id="test"
        )
        assert graph is not None
        nodes = graph.get_graph().nodes
        assert "load_memory" in nodes
        assert "summarize" in nodes
        assert "agent" in nodes
        assert "tools" in nodes
        assert "check_goals" in nodes
        assert "save_memory" in nodes

    def test_graph_edges_include_summarize(self, sample_df, mock_chatgroq):
        graph = build_graph(
            df=sample_df, api_key="fake_key", model="fake_model", user_id="test"
        )
        edges = graph.get_graph().edges
        edge_pairs = [(e.source, e.target) for e in edges]
        assert ("load_memory", "summarize") in edge_pairs
        assert ("summarize", "agent") in edge_pairs


class TestSummarizationConfig:
    def test_threshold_is_positive(self):
        assert SUMMARIZATION_THRESHOLD > 0

    def test_keep_recent_smaller_than_threshold(self):
        assert KEEP_RECENT_MESSAGES < SUMMARIZATION_THRESHOLD

    def test_keep_recent_is_even(self):
        """KEEP_RECENT debe ser par para preservar pares user/assistant completos."""
        assert KEEP_RECENT_MESSAGES % 2 == 0


class TestSummarizeNodeBehavior:
    """Tests del comportamiento del summarize_node via invocacion del grafo."""

    def test_short_conversation_skips_summarization(self, sample_df, mock_chatgroq, tmp_path):
        """Conversaciones cortas no deben activar el resumidor."""
        with patch("src.agent.graph.load_memory") as mock_load, \
             patch("src.agent.graph.save_memory") as mock_save:
            mock_load.return_value = {
                "goals": [], "past_alerts": [], "savings_tips_given": [], "summary": ""
            }
            graph = build_graph(
                df=sample_df, api_key="fake", model="fake", user_id="test"
            )

            short_messages = [HumanMessage(content="Hola")]
            graph.invoke({"messages": short_messages})

            # El LLM se llamo al menos una vez para responder
            instance = mock_chatgroq.return_value
            assert instance.invoke.called
            # Pero NO se llamo con un prompt de resumen (no hay SUMMARY_SYSTEM_PROMPT en los calls)
            calls_with_summary = [
                c for c in instance.invoke.call_args_list
                if any(
                    "resume" in str(getattr(m, "content", "")).lower()
                    for m in (c.args[0] if c.args else [])
                )
            ]
            assert len(calls_with_summary) == 0

    def test_long_conversation_triggers_summarization(self, sample_df, mock_chatgroq, tmp_path):
        """Conversaciones que superan el threshold deben activar el resumidor."""
        with patch("src.agent.graph.load_memory") as mock_load, \
             patch("src.agent.graph.save_memory") as mock_save:
            mock_load.return_value = {
                "goals": [], "past_alerts": [], "savings_tips_given": [], "summary": ""
            }
            graph = build_graph(
                df=sample_df, api_key="fake", model="fake", user_id="test"
            )

            # Crear una conversacion larga (mas que SUMMARIZATION_THRESHOLD)
            long_messages = []
            for i in range(SUMMARIZATION_THRESHOLD + 5):
                if i % 2 == 0:
                    long_messages.append(HumanMessage(content=f"Pregunta {i}"))
                else:
                    long_messages.append(AIMessage(content=f"Respuesta {i}"))

            graph.invoke({"messages": long_messages})

            instance = mock_chatgroq.return_value
            # Debe haber al menos una llamada con el prompt de resumen
            calls_with_summary = [
                c for c in instance.invoke.call_args_list
                if any(
                    "resume" in str(getattr(m, "content", "")).lower()
                    for m in (c.args[0] if c.args else [])
                )
            ]
            assert len(calls_with_summary) >= 1

    def test_summary_persisted_in_memory(self, sample_df, mock_chatgroq, tmp_path):
        """Tras resumir, save_memory debe recibir el resumen actualizado."""
        saved_memory = {}

        def fake_save(memory, user_id, base_dir=None):
            saved_memory.update(memory)
            return tmp_path / "fake.json"

        with patch("src.agent.graph.load_memory") as mock_load, \
             patch("src.agent.graph.save_memory", side_effect=fake_save):
            mock_load.return_value = {
                "goals": [], "past_alerts": [], "savings_tips_given": [], "summary": ""
            }
            graph = build_graph(
                df=sample_df, api_key="fake", model="fake", user_id="test"
            )

            long_messages = [
                HumanMessage(content=f"msg {i}") if i % 2 == 0
                else AIMessage(content=f"resp {i}")
                for i in range(SUMMARIZATION_THRESHOLD + 5)
            ]
            graph.invoke({"messages": long_messages})

            assert "summary" in saved_memory
            assert saved_memory["summary"] == "Resumen mockeado"