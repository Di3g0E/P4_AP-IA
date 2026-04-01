"""Motor de analytics financiero: resumenes, tendencias, anomalias."""

import numpy as np
import pandas as pd


def compute_monthly_summary(
    df: pd.DataFrame, year: int | None = None, month: int | None = None
) -> dict:
    """Calcula resumen de un mes: ingresos, gastos, ahorro, breakdown por Area.

    Si no se especifica year/month, usa el mes completo mas reciente.
    """
    if year and month:
        mask = (df["Year"] == year) & (df["Month"] == month)
        month_df = df[mask]
    else:
        # Ultimo mes con datos
        latest_period = df["YearMonth"].max()
        month_df = df[df["YearMonth"] == latest_period]

    expenses = month_df[month_df["Type"] == "Expenses"]
    income = month_df[month_df["Type"] == "Income"]

    total_inc = income["Amount_clean"].sum()
    total_exp = expenses["Amount_clean"].sum()

    breakdown = (
        expenses.groupby("Area")["Amount_clean"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "period": str(month_df["YearMonth"].iloc[0]) if not month_df.empty else "N/A",
        "income": total_inc,
        "expenses": total_exp,
        "net_savings": total_inc - total_exp,
        "savings_rate": (total_inc - total_exp) / total_inc * 100 if total_inc > 0 else 0,
        "n_transactions": len(month_df),
        "breakdown": breakdown,
    }


def compute_spending_trends(df: pd.DataFrame, n_months: int = 6) -> dict:
    """Calcula tendencias de gasto mensual: totales y cambios porcentuales."""
    expenses = df[df["Type"] == "Expenses"]
    monthly_totals = (
        expenses.groupby("YearMonth")["Amount_clean"]
        .sum()
        .sort_index(ascending=False)
        .head(n_months)
    )

    # Cambio porcentual mes a mes
    pct_changes = monthly_totals.pct_change(periods=-1) * 100

    # Tendencia por categoria
    recent_periods = monthly_totals.index[:n_months]
    category_trends = {}
    for area in expenses["Area"].unique():
        area_monthly = (
            expenses[expenses["Area"] == area]
            .groupby("YearMonth")["Amount_clean"]
            .sum()
        )
        area_recent = area_monthly[area_monthly.index.isin(recent_periods)]
        if len(area_recent) >= 2:
            first_half = area_recent.iloc[len(area_recent) // 2:].mean()
            second_half = area_recent.iloc[:len(area_recent) // 2].mean()
            if first_half > 0:
                change = (second_half - first_half) / first_half * 100
                category_trends[area] = round(change, 1)

    return {
        "monthly_totals": {str(k): round(v, 2) for k, v in monthly_totals.items()},
        "pct_changes": {str(k): round(v, 1) for k, v in pct_changes.dropna().items()},
        "category_trends": category_trends,
    }


def compute_category_breakdown(
    df: pd.DataFrame, period: str | None = None
) -> dict:
    """Calcula desglose de gastos por Area: absoluto y porcentaje.

    period: 'YYYY-MM' para un mes especifico, None para todo el historico.
    """
    expenses = df[df["Type"] == "Expenses"]
    if period:
        expenses = expenses[expenses["YearMonth"] == pd.Period(period, freq="M")]

    total = expenses["Amount_clean"].sum()
    breakdown = expenses.groupby("Area")["Amount_clean"].agg(["sum", "count", "mean"])
    breakdown = breakdown.sort_values("sum", ascending=False)
    breakdown["pct"] = breakdown["sum"] / total * 100 if total > 0 else 0

    return {
        "total_expenses": round(total, 2),
        "categories": {
            area: {
                "total": round(row["sum"], 2),
                "count": int(row["count"]),
                "mean": round(row["mean"], 2),
                "pct": round(row["pct"], 1),
            }
            for area, row in breakdown.iterrows()
        },
    }


def compute_savings_rate(df: pd.DataFrame, n_months: int = 6) -> dict:
    """Calcula tasa de ahorro mensual: (ingreso - gasto) / ingreso."""
    monthly = (
        df.groupby(["YearMonth", "Type"])["Amount_clean"]
        .sum()
        .unstack(fill_value=0)
    )

    if "Income" not in monthly.columns:
        monthly["Income"] = 0
    if "Expenses" not in monthly.columns:
        monthly["Expenses"] = 0

    monthly = monthly.sort_index(ascending=False).head(n_months)
    monthly["savings"] = monthly["Income"] - monthly["Expenses"]
    monthly["rate"] = np.where(
        monthly["Income"] > 0,
        monthly["savings"] / monthly["Income"] * 100,
        0,
    )

    avg_rate = monthly["rate"].mean()

    return {
        "monthly_rates": {
            str(k): round(row["rate"], 1)
            for k, row in monthly.iterrows()
        },
        "average_rate": round(avg_rate, 1),
        "monthly_savings": {
            str(k): round(row["savings"], 2)
            for k, row in monthly.iterrows()
        },
    }


def detect_anomalies(df: pd.DataFrame) -> list[dict]:
    """Detecta gastos anomalos: transacciones > media + 1.5*std por categoria."""
    expenses = df[df["Type"] == "Expenses"]
    anomalies = []

    for area in expenses["Area"].unique():
        area_data = expenses[expenses["Area"] == area]["Amount_clean"]
        if len(area_data) < 5:
            continue

        mean_val = area_data.mean()
        std_val = area_data.std()
        threshold = mean_val + 1.5 * std_val

        area_anomalies = expenses[
            (expenses["Area"] == area) & (expenses["Amount_clean"] > threshold)
        ]

        for _, row in area_anomalies.iterrows():
            anomalies.append({
                "date": row["Date"],
                "description": row["Description"],
                "amount": round(row["Amount_clean"], 2),
                "area": area,
                "mean": round(mean_val, 2),
                "threshold": round(threshold, 2),
            })

    return sorted(anomalies, key=lambda x: x["amount"], reverse=True)


def compute_recurring_expenses(df: pd.DataFrame) -> list[dict]:
    """Detecta gastos recurrentes (suscripciones, facturas) por patron de descripcion."""
    expenses = df[df["Type"] == "Expenses"]
    keywords = [
        "Gimnasio", "Netflix", "Spotify", "Seguro", "Gas",
        "Internet", "Agua", "Telefono", "HBO", "Amazon Prime",
    ]

    recurring = []
    for kw in keywords:
        matches = expenses[expenses["Description"].str.contains(kw, case=False, na=False)]
        if len(matches) >= 2:
            recurring.append({
                "name": kw,
                "avg_amount": round(matches["Amount_clean"].mean(), 2),
                "n_payments": len(matches),
                "n_months": matches["YearMonth"].nunique(),
                "total": round(matches["Amount_clean"].sum(), 2),
            })

    return sorted(recurring, key=lambda x: x["avg_amount"], reverse=True)


def compute_investment_summary(df: pd.DataFrame) -> dict:
    """Resume inversiones: total por instrumento, media mensual."""
    investments = df[
        (df["Area"].str.contains("Investment", case=False, na=False))
        & (df["Type"] == "Expenses")
    ]

    if investments.empty:
        return {"total": 0, "instruments": {}, "monthly_avg": 0}

    # Extraer instrumentos de las descripciones
    instrument_keywords = {
        "Letras del Tesoro": ["letra", "tesoro"],
        "Acciones Apple": ["apple", "acciones de apple"],
        "S&P 500": ["s&p", "sp500", "s&p 500"],
        "Bitcoin": ["bitcoin", "btc"],
        "Plan de Pensiones": ["pension", "pensiones"],
        "Fondo de Inversion": ["fondo"],
    }

    instruments = {}
    for name, keywords in instrument_keywords.items():
        mask = investments["Description"].str.lower().apply(
            lambda d: any(kw in d for kw in keywords)
        )
        matches = investments[mask]
        if not matches.empty:
            instruments[name] = {
                "total": round(matches["Amount_clean"].sum(), 2),
                "n_operations": len(matches),
                "avg_amount": round(matches["Amount_clean"].mean(), 2),
            }

    total = investments["Amount_clean"].sum()
    n_months = investments["YearMonth"].nunique()
    monthly_avg = total / n_months if n_months > 0 else 0

    return {
        "total": round(total, 2),
        "instruments": instruments,
        "monthly_avg": round(monthly_avg, 2),
        "n_months": n_months,
    }
