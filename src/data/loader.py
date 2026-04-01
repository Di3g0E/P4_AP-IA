"""Modulo de carga y parseo de datos financieros."""

import pandas as pd
from pathlib import Path


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Carga el CSV de transacciones y devuelve un DataFrame limpio y tipado.

    Parsea importes en formato euro ('10,00€' -> 10.00) y fechas DD/MM/YYYY.
    Anade columnas derivadas: Year, Month, YearMonth.
    """
    df = pd.read_csv(path)

    # Parsear importes: "10,00€" -> 10.00, "1.000,50€" -> 1000.50
    df["Amount_clean"] = (
        df["Amount"]
        .str.replace("€", "", regex=False)
        .str.replace(".", "", regex=False)   # separador de miles
        .str.replace(",", ".", regex=False)  # separador decimal
        .astype(float)
    )

    # Parsear fechas
    df["Date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Columnas derivadas
    df["Year"] = df["Date_parsed"].dt.year
    df["Month"] = df["Date_parsed"].dt.month
    df["YearMonth"] = df["Date_parsed"].dt.to_period("M")

    # Ordenar por fecha descendente
    df = df.sort_values("Date_parsed", ascending=False).reset_index(drop=True)

    return df
