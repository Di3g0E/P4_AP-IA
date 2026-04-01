"""Tests para el modulo de carga de datos."""

import pandas as pd
import pytest
from pathlib import Path

from src.data.loader import load_transactions


# Ruta al CSV de test (datos reales de P2)
CSV_PATH = Path(__file__).resolve().parent.parent.parent / "P2_AP-IA" / "data" / "raw" / "db_mod_descript.csv"


@pytest.fixture
def df():
    return load_transactions(CSV_PATH)


def test_load_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_required_columns_exist(df):
    required = ["Description", "Date", "Amount", "Area", "Type",
                 "Amount_clean", "Date_parsed", "Year", "Month", "YearMonth"]
    for col in required:
        assert col in df.columns, f"Falta columna: {col}"


def test_amount_parsing(df):
    """Verifica que los importes se parsean correctamente a float."""
    assert df["Amount_clean"].dtype == float
    assert (df["Amount_clean"] > 0).all(), "Todos los importes deben ser positivos"


def test_date_parsing(df):
    """Verifica que las fechas se parsean correctamente."""
    assert pd.api.types.is_datetime64_any_dtype(df["Date_parsed"])
    # El dataset va de 2021 a 2026
    assert df["Date_parsed"].min().year >= 2021
    assert df["Date_parsed"].max().year <= 2026


def test_sorted_by_date_descending(df):
    """Verifica que las transacciones estan ordenadas de mas reciente a mas antigua."""
    dates = df["Date_parsed"].tolist()
    assert dates == sorted(dates, reverse=True)


def test_derived_columns(df):
    """Verifica columnas derivadas Year, Month, YearMonth."""
    assert df["Year"].dtype in (int, "int64", "int32")
    assert df["Month"].dtype in (int, "int64", "int32")
    assert all(1 <= m <= 12 for m in df["Month"])


def test_types_are_valid(df):
    """Verifica que Type solo contiene Income o Expenses."""
    valid_types = {"Income", "Expenses"}
    assert set(df["Type"].unique()).issubset(valid_types)


def test_known_amount_value(df):
    """Verifica un valor concreto conocido del CSV."""
    # Primera fila del CSV: "10,00€" -> 10.0
    row = df[df["Description"].str.contains("Cena y copas en Cinesa", na=False)].iloc[0]
    assert row["Amount_clean"] == 10.0
