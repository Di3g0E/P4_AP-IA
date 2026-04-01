"""Modulo de retrieval estructurado para transacciones financieras."""

import unicodedata
import re
from datetime import datetime

import pandas as pd


def _normalize_text(text: str) -> str:
    """Normaliza texto para busqueda: minusculas, sin acentos."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def get_transactions_by_area(
    df: pd.DataFrame, area: str, limit: int = 20
) -> pd.DataFrame:
    """Filtra transacciones por Area (busqueda parcial, case-insensitive)."""
    mask = df["Area"].str.contains(area, case=False, na=False)
    return df[mask].head(limit)


def get_transactions_by_date_range(
    df: pd.DataFrame,
    start: datetime | str,
    end: datetime | str,
) -> pd.DataFrame:
    """Filtra transacciones dentro de un rango de fechas."""
    if isinstance(start, str):
        start = pd.to_datetime(start, dayfirst=True)
    if isinstance(end, str):
        end = pd.to_datetime(end, dayfirst=True)
    mask = (df["Date_parsed"] >= start) & (df["Date_parsed"] <= end)
    return df[mask]


def get_recent_transactions(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Devuelve las N transacciones mas recientes."""
    return df.head(n)


def get_top_expenses(
    df: pd.DataFrame, n: int = 10, area: str | None = None
) -> pd.DataFrame:
    """Devuelve los N mayores gastos, opcionalmente filtrados por area."""
    expenses = df[df["Type"] == "Expenses"]
    if area:
        expenses = expenses[expenses["Area"].str.contains(area, case=False, na=False)]
    return expenses.nlargest(n, "Amount_clean")


def search_transactions(
    df: pd.DataFrame, query: str, limit: int = 15
) -> pd.DataFrame:
    """Busca transacciones por texto en la descripcion (normalizado)."""
    normalized_query = _normalize_text(query)
    mask = df["Description"].apply(
        lambda x: normalized_query in _normalize_text(str(x))
    )
    return df[mask].head(limit)


def format_transactions_for_prompt(df_subset: pd.DataFrame) -> str:
    """Convierte un subset de transacciones en texto para incluir en el prompt.

    Formato: una linea por transaccion, compacto y legible.
    """
    if df_subset.empty:
        return "(Sin transacciones relevantes)"

    lines = []
    for _, row in df_subset.iterrows():
        tipo = "Gasto" if row["Type"] == "Expenses" else "Ingreso"
        lines.append(
            f"- {row['Date']} | {row['Description']} | "
            f"{row['Amount_clean']:.2f} EUR | {row['Area']} | {tipo}"
        )
    return "\n".join(lines)
