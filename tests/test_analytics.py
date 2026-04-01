"""Tests para el motor de analytics financiero."""

import pandas as pd
import pytest
from pathlib import Path

from src.data.loader import load_transactions
from src.features.analytics import (
    compute_category_breakdown,
    compute_investment_summary,
    compute_monthly_summary,
    compute_recurring_expenses,
    compute_savings_rate,
    compute_spending_trends,
    detect_anomalies,
)


CSV_PATH = Path(__file__).resolve().parent.parent.parent / "P2_AP-IA" / "data" / "raw" / "db_mod_descript.csv"


@pytest.fixture
def df():
    return load_transactions(CSV_PATH)


class TestMonthlySummary:
    def test_returns_dict(self, df):
        result = compute_monthly_summary(df)
        assert isinstance(result, dict)

    def test_has_required_keys(self, df):
        result = compute_monthly_summary(df)
        for key in ["period", "income", "expenses", "net_savings", "savings_rate", "breakdown"]:
            assert key in result

    def test_expenses_positive(self, df):
        result = compute_monthly_summary(df)
        assert result["expenses"] >= 0

    def test_specific_month(self, df):
        result = compute_monthly_summary(df, year=2026, month=1)
        assert result["period"] == "2026-01"
        assert result["n_transactions"] > 0


class TestSpendingTrends:
    def test_returns_dict(self, df):
        result = compute_spending_trends(df)
        assert isinstance(result, dict)

    def test_has_monthly_totals(self, df):
        result = compute_spending_trends(df, n_months=3)
        assert "monthly_totals" in result
        assert len(result["monthly_totals"]) <= 3


class TestCategoryBreakdown:
    def test_returns_dict(self, df):
        result = compute_category_breakdown(df)
        assert isinstance(result, dict)

    def test_percentages_sum_to_100(self, df):
        result = compute_category_breakdown(df)
        total_pct = sum(c["pct"] for c in result["categories"].values())
        assert abs(total_pct - 100) < 1  # Tolerancia por redondeo

    def test_known_categories_present(self, df):
        result = compute_category_breakdown(df)
        categories = result["categories"]
        # Sabemos que existen Leisure, Food, Invoice en el dataset
        assert "Leisure" in categories or any("Leisure" in c for c in categories)


class TestSavingsRate:
    def test_returns_dict(self, df):
        result = compute_savings_rate(df)
        assert isinstance(result, dict)

    def test_has_average(self, df):
        result = compute_savings_rate(df)
        assert "average_rate" in result
        assert isinstance(result["average_rate"], float)


class TestAnomalies:
    def test_returns_list(self, df):
        result = detect_anomalies(df)
        assert isinstance(result, list)

    def test_anomaly_structure(self, df):
        result = detect_anomalies(df)
        if result:
            anomaly = result[0]
            assert "amount" in anomaly
            assert "area" in anomaly
            assert "mean" in anomaly
            assert anomaly["amount"] > anomaly["mean"]


class TestRecurringExpenses:
    def test_returns_list(self, df):
        result = compute_recurring_expenses(df)
        assert isinstance(result, list)

    def test_known_recurring(self, df):
        result = compute_recurring_expenses(df)
        names = [r["name"] for r in result]
        # Netflix y Gimnasio son recurrentes conocidos
        assert any("Netflix" in n for n in names) or any("Gimnasio" in n for n in names)


class TestInvestmentSummary:
    def test_returns_dict(self, df):
        result = compute_investment_summary(df)
        assert isinstance(result, dict)

    def test_has_total(self, df):
        result = compute_investment_summary(df)
        assert "total" in result
        assert result["total"] > 0  # Sabemos que hay inversiones en el dataset
