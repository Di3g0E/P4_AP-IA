"""Tools del agente financiero — wrappers sobre analytics/retrieval existentes + goals."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from src.features.analytics import (
    compute_category_breakdown,
    compute_monthly_summary,
    compute_recurring_expenses,
    compute_savings_rate,
    compute_spending_trends,
    detect_anomalies,
)
from src.data.retrieval import (
    get_top_expenses,
    format_transactions_for_prompt,
    search_transactions,
)

# --- DataFrame compartido (se inyecta al inicializar el agente) ---
_df: pd.DataFrame | None = None


def set_dataframe(df: pd.DataFrame) -> None:
    """Inyecta el DataFrame de transacciones para que las tools lo usen."""
    global _df
    _df = df


def _get_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("DataFrame no inicializado. Llama a set_dataframe() primero.")
    return _df


# --- Goals en memoria (se sincroniza con el estado del grafo) ---
_goals: list[dict] = []


def set_goals(goals: list[dict]) -> None:
    global _goals
    _goals = goals


def get_goals() -> list[dict]:
    return _goals


# =============================================================================
#  Tools de analytics (wrappers de funciones existentes)
# =============================================================================

@tool
def monthly_summary(year: Optional[int] = None, month: Optional[int] = None) -> str:
    """Obtiene el resumen financiero de un mes: ingresos, gastos, ahorro y desglose por categoria.
    Si no se indica mes, usa el mas reciente."""
    data = compute_monthly_summary(_get_df(), year, month)
    return json.dumps(data, ensure_ascii=False, default=str)


@tool
def spending_trends(n_months: int = 6) -> str:
    """Muestra tendencias de gasto mensual y por categoria en los ultimos N meses.
    Util para ver si el gasto sube o baja."""
    data = compute_spending_trends(_get_df(), n_months)
    return json.dumps(data, ensure_ascii=False, default=str)


@tool
def category_breakdown(period: Optional[str] = None) -> str:
    """Desglose de gastos por categoria con totales, porcentajes y medias.
    period: 'YYYY-MM' para un mes concreto, o None para todo el historico."""
    data = compute_category_breakdown(_get_df(), period)
    return json.dumps(data, ensure_ascii=False, default=str)


@tool
def savings_rate(n_months: int = 6) -> str:
    """Calcula la tasa de ahorro mensual (ingreso - gasto) / ingreso en los ultimos N meses."""
    data = compute_savings_rate(_get_df(), n_months)
    return json.dumps(data, ensure_ascii=False, default=str)


@tool
def anomalies() -> str:
    """Detecta gastos anomalamente altos (> media + 1.5*std por categoria)."""
    data = detect_anomalies(_get_df())
    return json.dumps(data[:10], ensure_ascii=False, default=str)


@tool
def recurring_expenses() -> str:
    """Lista gastos recurrentes detectados (suscripciones, facturas periodicas)."""
    data = compute_recurring_expenses(_get_df())
    return json.dumps(data, ensure_ascii=False, default=str)


@tool
def top_expenses(n: int = 10, area: Optional[str] = None) -> str:
    """Devuelve los N mayores gastos, opcionalmente filtrados por area/categoria."""
    df_result = get_top_expenses(_get_df(), n, area)
    return format_transactions_for_prompt(df_result)


@tool
def search_by_description(query: str) -> str:
    """Busca transacciones cuya descripcion contenga el texto indicado."""
    df_result = search_transactions(_get_df(), query)
    if df_result.empty:
        return "No se encontraron transacciones con esa descripcion."
    return format_transactions_for_prompt(df_result)


# =============================================================================
#  Tools de objetivos (goals) — nuevas
# =============================================================================

@tool
def set_goal(category: str, limit: float, deadline: str) -> str:
    """Establece un objetivo de gasto maximo mensual para una categoria.
    category: area de gasto (ej: 'Restauración', 'Ocio').
    limit: importe maximo en EUR al mes.
    deadline: mes limite en formato YYYY-MM (ej: '2026-06')."""
    global _goals
    # Actualizar si ya existe uno para esa categoria
    _goals = [g for g in _goals if g["category"].lower() != category.lower()]
    _goals.append({
        "category": category,
        "limit": limit,
        "deadline": deadline,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
    })
    return f"Objetivo registrado: maximo {limit:.2f} EUR/mes en {category} hasta {deadline}."


@tool
def check_goal_progress(category: str) -> str:
    """Comprueba el progreso de un objetivo para una categoria concreta.
    Devuelve gasto actual vs limite y porcentaje consumido."""
    df = _get_df()
    goal = next((g for g in _goals if g["category"].lower() == category.lower()), None)
    if not goal:
        return f"No hay objetivo definido para '{category}'."

    # Gasto del mes actual
    now = datetime.now()
    summary = compute_monthly_summary(df, now.year, now.month)
    spent = summary["breakdown"].get(category, 0)
    limit = goal["limit"]
    pct = (spent / limit * 100) if limit > 0 else 0

    status = "dentro del limite"
    if pct >= 100:
        status = "SUPERADO"
    elif pct >= 80:
        status = "cerca del limite"

    return (
        f"Objetivo '{category}': {spent:.2f} / {limit:.2f} EUR "
        f"({pct:.0f}%) — {status}."
    )


@tool
def list_goals() -> str:
    """Lista todos los objetivos de ahorro activos del usuario."""
    if not _goals:
        return "No hay objetivos definidos. Usa set_goal para crear uno."
    lines = []
    for g in _goals:
        lines.append(
            f"- {g['category']}: max {g['limit']:.2f} EUR/mes "
            f"(hasta {g['deadline']}, creado {g['created_at']})"
        )
    return "\n".join(lines)


@tool
def remove_goal(category: str) -> str:
    """Elimina un objetivo de ahorro para una categoria."""
    global _goals
    before = len(_goals)
    _goals = [g for g in _goals if g["category"].lower() != category.lower()]
    if len(_goals) < before:
        return f"Objetivo de '{category}' eliminado."
    return f"No se encontro objetivo para '{category}'."


# =============================================================================
#  Lista de todas las tools para bindear al LLM
# =============================================================================

ALL_TOOLS = [
    monthly_summary,
    spending_trends,
    category_breakdown,
    savings_rate,
    anomalies,
    recurring_expenses,
    top_expenses,
    search_by_description,
    set_goal,
    check_goal_progress,
    list_goals,
    remove_goal,
]
