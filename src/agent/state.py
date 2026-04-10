"""Definicion del estado del grafo LangGraph."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Goal(TypedDict):
    """Objetivo de ahorro del usuario."""
    category: str       # Area del gasto (e.g. "Restauración")
    limit: float        # Limite mensual en EUR
    deadline: str       # Fecha limite YYYY-MM
    created_at: str     # Fecha de creacion YYYY-MM-DD


class UserMemory(TypedDict, total=False):
    """Memoria persistente entre sesiones."""
    goals: list[Goal]
    past_alerts: list[str]
    savings_tips_given: list[str]
    summary: str  # Resumen acumulado de conversaciones previas


class AgentState(TypedDict):
    """Estado completo del grafo del agente."""
    # Conversacion (se acumulan con add_messages)
    messages: Annotated[list[BaseMessage], add_messages]

    # Memoria persistente (cargada/guardada en disco)
    memory: UserMemory

    # Alertas generadas en esta ejecucion
    current_alerts: list[str]
