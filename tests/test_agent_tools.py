"""Tests para las tools del agente financiero."""

import json
import pytest
import pandas as pd

from src.agent.tools import (
    ALL_TOOLS,
    set_dataframe,
    set_goals,
    get_goals,
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
)


@pytest.fixture(autouse=True)
def sample_df():
    """DataFrame de ejemplo con transacciones minimas."""
    data = {
        "Date": ["01/01/2026", "15/01/2026", "01/02/2026", "15/02/2026", "01/03/2026"],
        "Description": ["Supermercado Dia", "Restaurante Sol", "Gimnasio", "Netflix", "Supermercado Dia"],
        "Amount": ["50,00€", "30,00€", "40,00€", "15,00€", "55,00€"],
        "Type": ["Expenses", "Expenses", "Expenses", "Expenses", "Expenses"],
        "Area": ["Alimentacion", "Restauracion", "Ocio", "Ocio", "Alimentacion"],
    }
    df = pd.DataFrame(data)
    df["Amount_clean"] = df["Amount"].str.replace("€", "").str.replace(",", ".").astype(float)
    df["Date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Year"] = df["Date_parsed"].dt.year
    df["Month"] = df["Date_parsed"].dt.month
    df["YearMonth"] = df["Date_parsed"].dt.to_period("M")
    df = df.sort_values("Date_parsed", ascending=False).reset_index(drop=True)
    set_dataframe(df)
    set_goals([])
    return df


class TestAnalyticsTools:
    def test_monthly_summary_returns_valid_json(self):
        result = monthly_summary.invoke({"year": 2026, "month": 1})
        data = json.loads(result)
        assert "income" in data
        assert "expenses" in data
        assert data["expenses"] > 0

    def test_spending_trends_returns_valid_json(self):
        result = spending_trends.invoke({"n_months": 3})
        data = json.loads(result)
        assert "monthly_totals" in data

    def test_category_breakdown_returns_valid_json(self):
        result = category_breakdown.invoke({})
        data = json.loads(result)
        assert "total_expenses" in data
        assert "categories" in data
        assert len(data["categories"]) > 0

    def test_savings_rate_returns_valid_json(self):
        result = savings_rate.invoke({"n_months": 3})
        data = json.loads(result)
        assert "average_rate" in data

    def test_anomalies_returns_list(self):
        result = anomalies.invoke({})
        data = json.loads(result)
        assert isinstance(data, list)

    def test_top_expenses_returns_text(self):
        result = top_expenses.invoke({"n": 3})
        assert isinstance(result, str)

    def test_search_by_description_finds_match(self):
        result = search_by_description.invoke({"query": "Supermercado"})
        assert "Supermercado" in result

    def test_search_by_description_no_match(self):
        result = search_by_description.invoke({"query": "xyznonexistent"})
        assert "No se encontraron" in result


class TestGoalTools:
    def test_set_goal_creates_goal(self):
        result = set_goal.invoke({"category": "Ocio", "limit": 100.0, "deadline": "2026-06"})
        assert "Objetivo registrado" in result
        assert len(get_goals()) == 1
        assert get_goals()[0]["category"] == "Ocio"

    def test_set_goal_updates_existing(self):
        set_goal.invoke({"category": "Ocio", "limit": 100.0, "deadline": "2026-06"})
        set_goal.invoke({"category": "Ocio", "limit": 200.0, "deadline": "2026-12"})
        assert len(get_goals()) == 1
        assert get_goals()[0]["limit"] == 200.0

    def test_list_goals_empty(self):
        result = list_goals.invoke({})
        assert "No hay objetivos" in result

    def test_list_goals_with_goals(self):
        set_goal.invoke({"category": "Ocio", "limit": 100.0, "deadline": "2026-06"})
        result = list_goals.invoke({})
        assert "Ocio" in result

    def test_check_goal_progress_no_goal(self):
        result = check_goal_progress.invoke({"category": "Ocio"})
        assert "No hay objetivo" in result

    def test_check_goal_progress_with_goal(self):
        set_goal.invoke({"category": "Alimentacion", "limit": 200.0, "deadline": "2026-06"})
        result = check_goal_progress.invoke({"category": "Alimentacion"})
        assert "Alimentacion" in result
        assert "EUR" in result

    def test_remove_goal(self):
        set_goal.invoke({"category": "Ocio", "limit": 100.0, "deadline": "2026-06"})
        result = remove_goal.invoke({"category": "Ocio"})
        assert "eliminado" in result
        assert len(get_goals()) == 0

    def test_remove_goal_not_found(self):
        result = remove_goal.invoke({"category": "Inexistente"})
        assert "No se encontro" in result


class TestAllToolsList:
    def test_all_tools_count(self):
        assert len(ALL_TOOLS) == 12

    def test_all_tools_have_names(self):
        for tool in ALL_TOOLS:
            assert tool.name is not None
            assert len(tool.name) > 0
